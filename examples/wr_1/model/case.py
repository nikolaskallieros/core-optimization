"""Connect one WR-1 scenario to the reusable fuel-cycle interfaces."""
from pathlib import Path
from coupling.cases import CaseModel, CycleCase, load_configuration
from materials.inventory import IsotopicInventory
from runtime.paths import NUCLEAR_DATA, RUNS
from examples.wr_1 import WR1_ROOT
from .geometry import Design, make_case, mesh as make_mesh
from .calibration import make_surrogate, equivalence_bases, fitting_targets
from diffusion_surrogate.openmc_cylinders import build
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
        mesh=lambda design, case: make_mesh(design, case,
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
