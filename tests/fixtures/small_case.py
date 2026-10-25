"""Small sector using the existing WR-1 materials and native OpenMC transport interface."""
from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from geometry.channels import ChannelDesign as WR1Design
from examples.wr_1.model.case import study as wr1_study
from examples.wr_1.model.materials import fuel_recipe
from coupling.cases import load_configuration
from materials.composition import transport_atoms


def settings(root):
    """Read the retained small-cycle fixture settings.

    Parameters
    ----------
    root : str or pathlib.Path
        WR-1 example root supplying the canonical model definition.

    Returns
    -------
    dict
        Settings from the fixture-local JSON; root is retained only for interface
        compatibility.
    """
    return json.loads((Path(__file__).parent / "wr1/small_cycle.json").read_text())


@lru_cache(maxsize=2)
def definition(root):
    """Resolve the canonical WR-1 model definition for the fixture.

    Parameters
    ----------
    root : str or pathlib.Path
        WR-1 example root supplying the canonical model definition.

    Returns
    -------
    dict
        Resolved model data from scenario one of inputs/study.toml.
    """
    return load_configuration(Path(root) / "inputs/study.toml", 1).data("model")


@lru_cache(maxsize=2)
def template(root):
    """Extract a central-channel material template from WR-1.

    Parameters
    ----------
    root : str or pathlib.Path
        WR-1 example root supplying the canonical model definition.

    Returns
    -------
    case : dict
        Unborated source transport case.
    shells : list of dict
        Material shells at one central-channel axial position.
    """
    case = wr1_study(1, root=root).make_case(boron=0.)
    channel = next(r for r in case["regions"] if r.get("ring") == 0 and r["depletable"])
    z = channel["z_bounds_cm"]
    shells = [r for r in case["regions"] if r.get("ring") == 0 and r["z_bounds_cm"] == z]
    return case, shells


class SmallDesign(WR1Design):
    def __init__(self, root):
        """Initialize the small-core geometry and composition parameters.

        Parameters
        ----------
        root : str or pathlib.Path
            WR-1 example root supplying the canonical model definition.
        """
        nn.Module.__init__(self)
        spec = settings(root)
        channels = spec["channels"]
        for name, value in spec["design"].items():
            self.register_parameter(name, nn.Parameter(torch.tensor(value, dtype=torch.float64)))
        self.channel_ids = tuple(c["id"] for c in channels)
        self.register_buffer("ring", torch.tensor([c["ring"] for c in channels]))
        self.register_buffer("fuelled", torch.ones(len(channels), dtype=torch.bool))
        self.register_buffer("lattice_xy", torch.tensor(
            [[math.cos(math.radians(c["angle_deg"])), math.sin(math.radians(c["angle_deg"]))]
             for c in channels], dtype=torch.float64))
        self.axial_count = spec["axial_positions"]
        self.register_buffer("z_edges_mm", torch.arange(self.axial_count + 1, dtype=torch.float64)
                             * spec["assembly_height_mm"])
        _, shells = template(str(Path(root)))
        interfaces = tuple(r["pieces"][0]["outer_radius_cm"] * 10 for r in shells)
        self.reference_interfaces = (interfaces,) * len(channels)
        self.assembly_ids = tuple(f"{c['id']}/axial/{j}" for c in channels for j in range(self.axial_count))


def make_case(root, *, design_values=None, boron=None, sampling=None, fuel_states=None):
    """Build the small hexagonal-sector transport fixture.

    Parameters
    ----------
    root : str or pathlib.Path
        WR-1 example root supplying the canonical model definition.
    design_values : mapping or None, optional
        Complete fixture design overrides in their declared parameter units.
    boron : float or None, optional
        Boron-10 in mg/kg unborated moderator; None uses the fixture setting.
    sampling : mapping or None, optional
        Native particle/batch overrides; None uses fixture sampling.
    fuel_states : mapping or None, optional
        Region-labelled fuel atom densities in atom/b-cm; None uses fresh recipes.

    Returns
    -------
    dict
        Native case with geometry, materials, identities, symmetry accounting and response
        requests.
    """
    import openmc.data
    from diffusion_surrogate.geometry import circle_polygon_area

    root = Path(root)
    spec = settings(root)
    original, shells = template(str(root))
    case = deepcopy(original)
    design = SmallDesign(root)
    if design_values is not None:
        if set(design_values) != dict(design.named_parameters()).keys():
            raise ValueError("Incomplete small-core design")
        with torch.no_grad():
            for name, value in design_values.items():
                getattr(design, name).copy_(torch.tensor(value, dtype=torch.float64))
    mapped = design()
    xy = mapped["channel_xy_mm"].detach().numpy() / 10
    radii = [r.detach().numpy() / 10 for r in mapped["channel_interfaces_mm"]]
    z = design.z_edges_mm.numpy() / 10
    a = spec["apothem_cm"]
    polygon = np.array([[0., 0.], [a, -a / math.sqrt(3)], [a, a / math.sqrt(3)]])
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    outer = np.array([r[-1] for r in radii])
    if np.any(distance <= outer[:, None] + outer[None, :]) or np.any(xy[:, 0] + outer >= a):
        raise ValueError("Small-core channels overlap or cross the outer boundary")
    model = definition(root)
    material_spec = model["transport"]
    aliases = material_spec["nuclide_aliases"]
    recipe = fuel_recipe(model)
    boron = spec["initial_boron_mg_B10_per_kg"] if boron is None else float(boron)
    if not np.isfinite(boron) or boron < 0:
        raise ValueError("Invalid boron concentration")
    regions, identity, shapes = [], {}, []
    occupied = 0.
    for site, channel in enumerate(spec["channels"]):
        fraction, ring = channel["sector_fraction"], channel["ring"]
        multiplicity = round(6 * fraction)
        names = (["center"] if ring == 0 else
                 [f"outer_{j}" for j in range(site - 1, 6, 2)])
        for shell, prototype in enumerate(shells):
            outer_r = radii[site][shell]
            inner_r = radii[site][shell - 1] if shell else 0.
            area = circle_polygon_area(xy[site], outer_r, polygon)
            if inner_r:
                area -= circle_polygon_area(xy[site], inner_r, polygon)
            expected = fraction * math.pi * (outer_r**2 - inner_r**2)
            if not np.isclose(area, expected, rtol=1e-10, atol=1e-9):
                raise ArithmeticError("Sector channel fraction is inconsistent")
            occupied += area
            for slab, (lo, hi) in enumerate(zip(z[:-1], z[1:])):
                level = design.axial_count - 1 - slab
                entry = deepcopy(prototype)
                fuel = shell == 0
                label = (f"{channel['id']}/axial/{level}" if fuel else
                         f"{channel['id']}/shell/{shell}/axial/{level}")
                entry.update(label=label, ring=ring, physical_multiplicity=multiplicity,
                             orbit_site_ids=[channel["id"]], z_bounds_cm=[float(lo), float(hi)],
                             volume_cm3=area * (hi - lo), pieces=[{
                                 "site_id": channel["id"], "xy_cm": xy[site].tolist(),
                                 "inner_radius_cm": float(inner_r), "outer_radius_cm": float(outer_r)}])
                if fuel:
                    fresh = recipe.atom_densities(float(design.enrichment[ring].detach()))
                    inert = model["fuel"]["inert_atom_density"]
                    atoms = dict(inert)
                    for name, value in (fresh if fuel_states is None else fuel_states[label]).items():
                        atoms[name] = atoms.get(name, 0.) + value
                    volume = math.pi * outer_r**2 * (hi - lo)
                    hm_density = sum(fresh[n] * openmc.data.atomic_mass(n) for n in ("U235", "U238"))
                    entry.update(atom_density=transport_atoms(atoms, aliases), fresh_fuel_atom_density=fresh,
                                 inert_atom_density=inert, full_assembly_volume_cm3=volume,
                                 initial_hm_kg_per_assembly=hm_density * 1e24 * volume / openmc.data.AVOGADRO / 1000,
                                 physical_assembly_ids=[f"{name}/axial/{level}" for name in names])
                    for name in entry["physical_assembly_ids"]:
                        if name in identity:
                            raise ArithmeticError("Duplicated physical assembly ID")
                        identity[name] = len(regions)
                regions.append(entry)
        shapes.append({"site_id": channel["id"], "xy_cm": xy[site].tolist(),
                       "outer_radius_cm": float(radii[site][-1])})
    domain_area = a * a / math.sqrt(3)
    moderator = model["materials"]["moderator"]
    atoms = dict(moderator["atom_density"])
    if boron:
        n10 = boron * 1e-6 * moderator["density_g_cm3"] / openmc.data.atomic_mass("B10") * openmc.data.AVOGADRO / 1e24
        ratio = material_spec["boron"]["natural_atom_fractions"]
        atoms.update(B10=n10, B11=n10 * ratio["B11"] / ratio["B10"])
    for slab, (lo, hi) in enumerate(zip(z[:-1], z[1:])):
        regions.append({"label": f"moderator/axial/{design.axial_count - 1 - slab}",
                        "kind": "wr1_background", "z_bounds_cm": [float(lo), float(hi)],
                        "volume_cm3": (domain_area - occupied) * (hi - lo), "atom_density": transport_atoms(atoms, aliases),
                        "temperature_K": moderator["temperature_K"], "depletable": False, "void": False,
                        "thermal_scattering": material_spec["moderator_thermal_scattering"]})
    volume = domain_area * (z[-1] - z[0])
    physical_count = sum(round(6 * c["sector_fraction"]) for c in spec["channels"]) * design.axial_count
    if len(identity) != physical_count or not np.isclose(sum(r["volume_cm3"] for r in regions), volume, rtol=1e-12):
        raise ArithmeticError("Small-core identity or volume accounting failed")
    lower, upper = [0., -a / math.sqrt(3), float(z[0])], [a, a / math.sqrt(3), float(z[-1])]
    case.update(geometry={"symmetry": "c6", "symmetry_boundary_type": spec["symmetry_boundary_type"],
                          "apothem_cm": a, "lower_cm": lower, "upper_cm": upper, "channel_shapes": shapes,
                          "boundaries": ["vacuum", "reflective", "reflective", "reflective", *spec["axial_boundaries"]]},
                regions=regions, physical_assembly_to_region=identity, symmetry_multiplier=6,
                power_W=spec["power_W"], boron_mg_B10_per_kg=boron,
                nuclides=sorted({n for r in regions for n in r["atom_density"]}),
                design={name: p.detach().tolist() for name, p in design.named_parameters()},
                sampling=dict(spec["sampling"] if sampling is None else sampling),
                comparison_edges_cm=[np.linspace(lower[0], upper[0], 9).tolist(),
                                     np.linspace(lower[1], upper[1], 9).tolist(), z.tolist()],
                geometry_checks={"volume_cm3": volume, "independent_fuel_states": len(spec["channels"]) * design.axial_count,
                                 "physical_axial_assemblies": physical_count,
                                 "represented_channels": sum(c["sector_fraction"] for c in spec["channels"])})
    case["material_response"]["region_indices"] = [i for i, r in enumerate(regions) if r["depletable"]]
    case["boron_response"]["region_indices"] = [i for i, r in enumerate(regions) if r.get("thermal_scattering")]
    from examples.wr_1.build_geometry import parameterized_geometry as cad_geometry
    case["cad_geometry"] = cad_geometry(design, case).record()
    return case


def study(root):
    """Bind small-core factories to the shared cycle-case interface.

    Parameters
    ----------
    root : str or pathlib.Path
        WR-1 example root supplying the canonical model definition.

    Returns
    -------
    CycleCase
        Fixture workflow using the configured local nuclear-data and run paths.
    """
    from coupling.cases import CaseModel, CycleCase
    from runtime.paths import RUNS, NUCLEAR_DATA
    from materials.inventory import IsotopicInventory
    from examples.wr_1.model.calibration import make_surrogate, equivalence_bases, fitting_targets
    from geometry.diffusion import discretize
    from examples.wr_1.build_geometry import parameterized_geometry
    from examples.wr_1.model.materials import live_recipe, initialize_loading
    root = Path(root)
    config = settings(root)
    work = RUNS / "small_cycle/channel_burnup"
    data = Path(__file__).parent / "wr1"
    model_definition = definition(root)
    aliases = model_definition["transport"]["nuclide_aliases"]

    def inventory(values):
        """Bind a small-core design to the full depletion chain.

        Parameters
        ----------
        values : mapping
            Complete small-core design parameter values.

        Returns
        -------
        IsotopicInventory
            Full-chain inventory and live fresh-fuel recipe binding.
        """
        options = dict(model_definition["depletion"])
        options.update(depletion_flux_groups=config["depletion_flux_groups"],
                       depletion_flux_refinement_check=False, sampling=config["sampling"])
        recipe = fuel_recipe(model_definition)
        return IsotopicInventory(case_factory=lambda **kwargs: make_case(root, **kwargs),
            recipe=lambda physical, design: live_recipe(physical, design, recipe, aliases),
            settings=options, design_values=values, aliases=aliases,
            chain_path=NUCLEAR_DATA / "wr_1/chain_endfb71_pwr.xml",
            cross_sections=NUCLEAR_DATA / "wr_1/depletion_cross_sections.xml",
            kernel_cache=work / "rate_kernel.npz")

    mesh = config["diffusion_mesh"]
    model = CaseModel(design=lambda: SmallDesign(root), inventory=inventory,
                      transport=lambda **kwargs: make_case(root, **kwargs),
                      surrogate=lambda design, state, layout, *, mesh: make_surrogate(model_definition, design,
                          state["case"], state["result"],
                          mesh=mesh,
                          equivalence_layout=layout, equivalence_void_flux=False),
                      equivalence_bases=equivalence_bases, fitting_targets=fitting_targets,
                      initialize_loading=initialize_loading,
                      mesh=lambda design, case: discretize(lambda: parameterized_geometry(design, case),
                          subdivisions=mesh["triangular_subdivisions"],
                          axial_subdivisions=mesh["axial_subdivisions_per_assembly"]))
    return CycleCase(name="small", model=model,
                     inputs={kind: data / ("small_cycle" + ("" if kind == "cycle" else f"_{kind}") + ".json")
                             for kind in ("cycle", "calibration", "optimization")},
                     results=work / "results", work=work, archive_root=work,
                     result_prefix="small_cycle_", data_subdirectory="")
