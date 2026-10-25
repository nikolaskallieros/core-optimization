"""WR-1 spatial closure groups, observation weights and mesh selection."""
import torch
from diffusion_surrogate import SparseDiffusion3D
from diffusion_surrogate.regional import RegionalSurrogate
from diffusion_surrogate.equivalence import equivalence_bases as spatial_bases, shared_bases
from diffusion_surrogate.solver import tensor
from optimization.fitting import native_targets
from materials.composition import map_nuclides
from .materials import fuel_recipe, fuel_volumes


def make_surrogate(definition, design, case, reference, *, mesh,
                   closure="equivalence", equivalence_layout="region", equivalence_void_flux=True):
    """Bind WR-1 material recipes and geometry to a regional diffusion surrogate.

    Parameters
    ----------
    definition : dict
        Resolved physical model from ``inputs/model.toml``, supplied by the case adapter.
    design : Design
        Live WR-1 design with differentiable ring and optional graphite parameters.
    case : dict
        Native region, material and symmetry description returned by ``make_case``.
    reference : dict
        Extracted native OpenMC responses, block statistics and cross sections.
    mesh : DiffusionGeometry
        Geometry selected by CaseModel: the native physical layout by default, or an
        explicitly supplied alternative retaining the same region identities and volumes.
    closure : {'shared', 'equivalence'}, optional
        Calibration family used to correct the measured native response.
    equivalence_layout : str, optional
        Regional equivalence grouping passed to ``equivalence_bases``.
    equivalence_void_flux : bool, optional
        Whether void regions receive independent reconstructed-flux factors.

    Returns
    -------
    RegionalSurrogate
        Differentiable regional model with fixed native reference data and live design
        inputs.
    """
    if closure == "shared":
        moderator = [i for i, region in enumerate(case["regions"]) if region.get("thermal_scattering")]
        names, bases = shared_bases(len(case["regions"]), len(case["group_edges_eV"]) - 1,
                                    moderator, selected_group=1)
        equivalence = None
    elif closure == "equivalence":
        names, bases, equivalence = equivalence_bases(case, layout=equivalence_layout,
                                                     include_void=equivalence_void_flux)
    else:
        raise ValueError(f"Unknown closure: {closure}")
    recipe = fuel_recipe(definition)
    aliases = definition["transport"]["nuclide_aliases"]

    def fresh(design, names, indices):
        """Evaluate fresh isotope densities in the requested fuel regions.

        Parameters
        ----------
        design : Design
            Live WR-1 design with differentiable ring and optional graphite parameters.
        names : sequence of str
            Transport nuclide names in the requested output column order.
        indices : sequence of int
            Indices of fuel regions in the native case.

        Returns
        -------
        torch.Tensor
            Fuel-region-by-nuclide atom densities in atoms/(barn cm), differentiable with
            respect to enrichment.
        """
        rings = torch.tensor([case["regions"][i]["ring"] for i in indices])
        atoms = map_nuclides(recipe.atom_densities(design.enrichment[rings]), aliases)
        return torch.stack([atoms[name] for name in names], dim=1)

    return RegionalSurrogate(design, case, reference,
                             geometry=mesh.fractions, solver=SparseDiffusion3D(mesh.grid),
                             names=names, bases=bases, equivalence=equivalence,
                             fuel_volumes=lambda: fuel_volumes(design, case),
                             aliases=aliases, fresh_densities=fresh,
                             geometry_key=mesh.identity)


def equivalence_bases(case, *, layout="region", include_void=True):
    """Construct independent regional flux and diffusion correction bases.

    Parameters
    ----------
    case : dict
        Native region, material and symmetry description returned by ``make_case``.
    layout : str, optional
        Grouping: region, material, ring_material or fuel_levels; an _axial suffix adds an
        axial mode.
    include_void : bool, optional
        Whether void regions participate in fitted flux reconstruction.

    Returns
    -------
    tuple
        Coefficient names, physical response bases and regional flux-equivalence bases.

    Notes
    -----
    Central axial moderator factors stay one to fix each group's amplitude. Group diffusion
    factors are fitted against flux and leakage. The shared solver's vacuum extrapolation is
    fixed: fitting it as well introduces two exact null directions when every regional flux
    correction is independent. Excluding void factors restricts the fit to coefficients that
    affect the transport equations; empty-gap flux then has no fitted reconstruction scale.
    See https://doi.org/10.1080/00223131.2011.649083 and the INL VTB Griffin/SPH
    formulation: https://virtualtestbed.inl.gov/sfr/vtr/vtr_model.html
    """
    regions, groups = case["regions"], len(case["group_edges_eV"]) - 1
    axial = layout.endswith("_axial")
    layout = layout.removesuffix("_axial")
    backgrounds = [r["label"] for r in regions if r["kind"] == "wr1_background"]
    anchor = backgrounds[len(backgrounds) // 2]
    if layout == "region":
        assignments = [None if r["label"] == anchor or (not include_void and r.get("void", False))
                       else r["label"] for r in regions]
    elif layout in ("material", "ring_material", "fuel_levels"):
        assignments = []
        for region in regions:
            if region["kind"] == "wr1_background" or (not include_void and region.get("void", False)):
                assignments.append(None)
                continue
            material = region.get("equivalence_material") or ("fuel" if region["depletable"] else
                       "shell/" + region["label"].split("/shell/")[1].split("/")[0])
            if layout == "fuel_levels":
                material = (f"ring/{region['ring']}/fuel/axial/{region['label'].rpartition('/axial/')[2]}"
                            if region["depletable"] else material)
                assignments.append(material)
            else:
                assignments.append(material if layout == "material" or material == "graphite"
                                   else f"ring/{region['ring']}/{material}")
    else:
        raise ValueError(f"Unknown equivalence layout: {layout}")
    coordinate = None
    if axial:
        lower, upper = case["geometry"]["lower_cm"][2], case["geometry"]["upper_cm"][2]
        z = tensor([sum(region["z_bounds_cm"]) / 2 for region in regions])
        coordinate = (z - (lower + upper) / 2) / ((upper - lower) / 2)
    return spatial_bases(assignments, groups, axial_coordinates=coordinate,
                         axial_mask=None if include_void else [not r.get("void", False) for r in regions])


def fitting_targets(reference, case):
    """Select and weight native observables for WR-1 calibration.

    Parameters
    ----------
    reference : dict
        Extracted native OpenMC responses, block statistics and cross sections.
    case : dict
        Native region, material and symmetry description returned by ``make_case``.

    Returns
    -------
    tuple
        Calibration Target objects and a matching reference mapping for criticality, fuel
        power, regional flux, capture, scattering and leakage, with declared relative noise
        floors.
    """
    fuel = [i for i, region in enumerate(case["regions"]) if region["depletable"]]
    material = [i for i, region in enumerate(case["regions"]) if not region["void"]]
    choices = [("keff", "keff", None, .001, 2.),
               ("fuel_power_W", "region_power_W", fuel, .03, 1.),
               ("region_flux_per_source", "region_flux_per_source", None, .05, 1.),
               ("capture_per_source", "capture_per_source", material, .05, 1.),
               ("scatter_per_source", "scatter_per_source", material, .05, 1.),
               ("leakage_per_source", "leakage_per_source", None, .05, 1.)]
    return native_targets(reference, choices)
