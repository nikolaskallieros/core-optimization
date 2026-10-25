"""Connect one WR-1 scenario to the reusable fuel-cycle interfaces."""
from pathlib import Path
from coupling.cases import CaseModel, CycleCase, load_configuration
from materials.inventory import IsotopicInventory
from runtime.paths import NUCLEAR_DATA, RUNS
from designs.wr_1 import WR1_ROOT
import math
import numpy as np
from materials.composition import transport_atoms
from geometry.channels import HexagonalDomain, channel_regions, cylinder_clearances
from geometry.diffusion import discretize
from .design import Design
from ..build_geometry import parameterized_geometry
from .calibration import make_surrogate, equivalence_bases, fitting_targets
from coupling.openmc import build
from .materials import fuel_recipe, live_recipe, initialize_loading


def study(step=1, *, root=WR1_ROOT, results=None, work=None):
    """Resolve one WR-1 scenario and bind it to the shared case interfaces.

    Parameters
    ----------
    step : int, optional
        Optimization scenario number, from 1 through 6.
    root : str or pathlib.Path, optional
        WR-1 example directory containing the authoritative inputs and model sources.
    results : str or pathlib.Path or None, optional
        Result directory; None selects the retained records for the requested scenario.
    work : str or pathlib.Path or None, optional
        Working directory; None selects the scenario directory under the configured runs
        root.

    Returns
    -------
    CycleCase
        Case adapter carrying the resolved configuration, model factories and explicit
        artifact paths.

    Notes
    -----
    study.toml declares model.toml and the two nuclear-data source manifests. They are
    loaded together once; geometry and material factories consume the resolved dictionaries.
    """
    root = Path(root)
    configuration = load_configuration(root / "inputs/study.toml", step)
    definition = configuration.data("model")
    settings = configuration.settings("cycle")
    aliases = definition["transport"]["nuclide_aliases"]
    mesh = settings["diffusion_mesh"]
    work = Path(work) if work is not None else RUNS / "wr_1" / f"optimization_step_{step}"
    archive = root / "results/wr1_optimization"
    results = Path(results) if results is not None else archive / f"optimization_step_{step}"
    transport = lambda **values: make_case(definition, settings, **values)

    def inventory(values):
        """Construct the physical isotope inventory for the resolved scenario.

        Parameters
        ----------
        values : dict or None
            Design values used to initialize the native case; None uses the scenario defaults.

        Returns
        -------
        IsotopicInventory
            Physical fuel inventory with its fresh recipe, depletion chain and rate-kernel cache
            paths.
        """
        options = dict(definition["depletion"])
        options.update(depletion_flux_groups=settings["depletion_flux_groups"], sampling=settings["sampling"])
        manifest = configuration.data("depletion_sources")
        directory = NUCLEAR_DATA / ("graphite" if "graphite" in settings else "wr_1")
        recipe = fuel_recipe(definition)
        return IsotopicInventory(case_factory=transport,
            recipe=lambda physical, design: live_recipe(physical, design, recipe, aliases),
            settings=options, design_values=values, aliases=aliases,
            physical_inventory_response=settings["physical_inventory_response"],
            chain_path=NUCLEAR_DATA / "wr_1/chain_endfb71_pwr.xml",
            chain_sha256=manifest["chain_sha256"],
            cross_sections=directory / "depletion_cross_sections.xml",
            kernel_cache=work / "rate_kernel.npz")

    model = CaseModel(
        design=lambda: Design(definition, settings), inventory=inventory, transport=transport,
        surrogate=lambda design, state, layout, *, mesh: make_surrogate(
            definition, design, state["case"], state["result"],
            mesh=mesh,
            equivalence_layout=layout, equivalence_void_flux=False),
        equivalence_bases=equivalence_bases, fitting_targets=fitting_targets,
        initialize_loading=initialize_loading, export_transport=build,
        mesh=lambda design, case: discretize(lambda: parameterized_geometry(design, case),
            subdivisions=mesh["triangular_subdivisions"],
            axial_subdivisions=mesh["axial_subdivisions_per_assembly"]))
    return CycleCase(name="wr1", configuration=configuration,
                     results=results, work=work, archive_root=archive, model=model,
                     initialize_loading="graphite" not in settings)


def new_run(step=1):
    """Bind a scenario to new-run outputs while retaining historical records.

    Parameters
    ----------
    step : int, optional
        Optimization scenario number, from 1 through 6.

    Returns
    -------
    CycleCase
        Case adapter whose results live under the configured runs directory.
    """
    return study(step, results=RUNS / "wr_1" / f"optimization_step_{step}" / "results")


def make_case(definition, scenario, *, design_values=None, boron=None, sampling=None, fuel_states=None):
    """Build a native WR-1 case from resolved model and scenario inputs.

    Parameters
    ----------
    definition : dict
        Resolved physical model from ``inputs/model.toml``, supplied by the case adapter.
    scenario : dict
        Resolved cycle settings from ``inputs/study.toml`` after scenario overrides.
    design_values : dict or None, optional
        Replacement design values; None uses the scenario's initial design.
    boron : float or None, optional
        B-10 concentration in mg per kg of unborated moderator; None uses the scenario
        value.
    sampling : dict or None, optional
        OpenMC particle, batch, inactive-batch, seed and thread settings; None uses the
        scenario settings.
    fuel_states : dict or None, optional
        Nuclide atom densities in atoms/(barn cm), keyed by representative assembly
        identity. None creates fresh fuel.

    Returns
    -------
    dict
        Native materials, regions, physical assembly mapping, tally settings, design values
        and geometry checks for the selected symmetry domain.

    Raises
    ------
    ValueError
        Geometry overlaps, violates symmetry or boundaries, or has invalid boron or design
        values.
    ArithmeticError
        The physical assembly-to-region map is incomplete.

    Notes
    -----
    The definition dictionary originates in model.toml and the scenario dictionary in
    study.toml. The case adapter resolves both; this function performs no configuration-file
    reads.
    """
    import openmc.data
    from diffusion_surrogate.symmetry import audit_c6_orbits
    from optimization.design import set_parameters

    spec = definition["transport"]
    aliases = spec["nuclide_aliases"]
    recipe = fuel_recipe(definition)
    design = Design(definition, scenario)
    if design_values is not None:
        set_parameters(design, design_values)
    mapped = design()
    centers = mapped["channel_xy_mm"].detach().numpy() / 10
    radii = [value.detach().numpy() / 10 for value in mapped["channel_interfaces_mm"]]
    fuelled, ring = design.fuelled.numpy(), design.ring.numpy()
    labels = [(bool(f), tuple(r), float(design.enrichment[ring[i]].detach()) if f else None)
              for i, (f, r) in enumerate(zip(fuelled, radii))]
    orbits = audit_c6_orbits(centers, labels, tolerance_cm=1e-9)
    identities = {i: min(design.channel_ids[j] for j in orbit) for orbit in orbits for i in orbit}
    domain = HexagonalDomain(scenario["apothem_cm"])
    clearance = cylinder_clearances(domain, centers, [r[-1] for r in radii])
    if min(clearance.values()) <= 0:
        raise ValueError("Channel interfaces overlap or cross the outer moderator boundary")
    if scenario["outer_radial_boundary"] not in ("vacuum", "reflective"):
        raise ValueError("The outer radial boundary must be vacuum or reflective")
    z = design.z_edges_mm.numpy() / 10
    boron = scenario["initial_boron_mg_B10_per_kg"] if boron is None else float(boron)
    if boron < 0 or not math.isfinite(boron):
        raise ValueError("Boron concentration must be finite and nonnegative")

    def material(name):
        """Resolve one material composition and apply moderator boron.

        Parameters
        ----------
        name : str
            Material key in the resolved physical model.

        Returns
        -------
        dict
            Transport-compatible atom densities in atoms/(barn cm) and temperature in K.
        """
        entry = definition["materials"][name]
        atoms = dict(entry["atom_density"])
        if name == "moderator" and boron:
            n10 = boron * 1e-6 * entry["density_g_cm3"] / openmc.data.atomic_mass("B10") * openmc.data.AVOGADRO / 1e24
            ratio = spec["boron"]["natural_atom_fractions"]
            atoms.update(B10=n10, B11=n10 * ratio["B11"] / ratio["B10"])
        return {"atom_density": transport_atoms(atoms, aliases), "temperature_K": entry["temperature_K"]}

    def shell_material(site, shell, level, volume):
        """Assign fresh or depleted material to one channel shell.

        Parameters
        ----------
        site : int
            Physical channel index in the lattice definition.
        shell : int
            Radial shell index; zero denotes the homogenized fuel element in a fuelled channel.
        level : int
            Axial assembly index, counted from the top.
        volume : float
            Volume of one physical shell segment in cm³.

        Returns
        -------
        dict
            Material composition, temperature and depletion metadata, including fresh
            heavy-metal mass for fuel.
        """
        if fuelled[site] and shell == 0:
            fresh = recipe.atom_densities(float(design.enrichment[ring[site]].detach()))
            inert = definition["fuel"]["inert_atom_density"]
            fuel = fresh if fuel_states is None else fuel_states[f"{identities[site]}/axial/{level}"]
            atoms = dict(inert)
            for nuclide, number in fuel.items():
                atoms[nuclide] = atoms.get(nuclide, 0.) + number
            hm_density = sum(fresh[n] * openmc.data.atomic_mass(n) for n in ("U235", "U238")) * 1e24 / openmc.data.AVOGADRO / 1000
            return {"atom_density": transport_atoms(atoms, aliases), "temperature_K": spec["homogenized_element_temperature_K"],
                    "depletable": True, "fresh_fuel_atom_density": fresh, "inert_atom_density": inert,
                    "initial_hm_kg_per_assembly": hm_density * volume}
        binding = design.channel_materials[site][shell]
        return {**material(binding), "depletable": False,
                **({"thermal_scattering": spec["moderator_thermal_scattering"]} if binding == "moderator" else {})}

    background = {**material("moderator"), "label": "moderator", "kind": "wr1_background",
                  "depletable": False, "thermal_scattering": spec["moderator_thermal_scattering"], "void": False}
    case = channel_regions(domain, design.channel_ids, centers, radii, z, orbits, shell_material, background,
                           site_metadata=[{"ring": int(value)} for value in ring], region_kind="wr1_shell")
    if len(case["physical_assembly_to_region"]) != int(fuelled.sum()) * design.axial_count:
        raise ArithmeticError("The WR-1 assembly identity map is incomplete")
    regions = case["regions"]
    lower, upper = domain.bounds(z)
    nx, ny = spec["comparison_mesh_subdivisions"]
    case.update({
        "power_W": scenario["power_W"], "symmetry_multiplier": domain.multiplier,
        "boron_mg_B10_per_kg": boron, "nuclides": sorted({name for entry in regions for name in entry["atom_density"]}),
        "design": {name: value.detach().tolist() for name, value in design.named_parameters()},
        "sampling": dict(scenario["sampling"] if sampling is None else sampling),
        "statistics_block_size": spec["statistics_block_size"], "group_edges_eV": spec["group_edges_eV"],
        "comparison_edges_cm": [np.linspace(lower[0], upper[0], nx + 1).tolist(),
                                np.linspace(lower[1], upper[1], ny + 1).tolist(), z.tolist()],
        "cross_sections": str(NUCLEAR_DATA / "wr_1/cross_sections.xml"),
        "temperature_label_tolerance_K": spec["temperature_label_tolerance_K"],
        "full_observations": True, "observation_cross_sections": "macroscopic_tallies",
        "need_depletion_rates": False, "depletion_rate_estimator": "native_microscopic_tracklength_for_all_chain_parents_with_transport_data",
        "depletion_nuclides": [], "depletion_reactions": [],
        "material_response": {"region_indices": [i for i, r in enumerate(regions) if r["depletable"]],
                              "nuclides": spec["material_response_nuclides"]},
        "boron_response": {"region_indices": [i for i, r in enumerate(regions) if r.get("thermal_scattering")],
                           "nuclides": list(spec["boron"]["natural_atom_fractions"])},
        "effective_approximations": {name: spec[name] for name in ("temperature_basis", "carbon_transport", "oxygen_transport", "organic_thermal_scattering")},
    })
    case["geometry"].update(symmetry_boundary_type=scenario["symmetry_boundary_type"],
                            boundaries=[scenario["outer_radial_boundary"], "reflective", "reflective", "reflective",
                                        *scenario["axial_boundaries"]])
    if "graphite" in scenario:
        if scenario["graphite"].get("mode") == "assembly_sleeves":
            from .sleeves import add_sleeves as add_graphite
        else:
            from .graphite import add_graphite
        case = add_graphite(case, design, mapped, NUCLEAR_DATA / "graphite/cross_sections.xml")
    case["cad_geometry"] = parameterized_geometry(design, case).record()
    return case
