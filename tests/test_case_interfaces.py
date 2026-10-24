"""A second case uses the shared interfaces without WR-1 geometry or parameters."""

import json
import signal
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from coupling.cases import CaseModel, CycleCase
from fuel_cycle.residence import DAY, FuelMap, replay_history
from coupling.execution import prepare_case, run_case
from optimization.design import parameters, set_parameters
from coupling.optimization import project
from diffusion_surrogate.equivalence import equivalence_bases
from diffusion_surrogate.inventory import InventoryCrossSections
from runtime.processes import interrupt_on_sigterm


def spatial_slab_case(tmp_path):
    """Extend the independent case with two material layers and a shared mesh.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated input and output directory supplied by pytest.

    Returns
    -------
    CycleCase
        Native layered geometry and a material-calibrated diffusion factory.
    """
    from diffusion_surrogate import CalibratedDiffusion, Diffusion3D, DiffusionGeometry, Grid3D
    from diffusion_surrogate.solver import tensor

    case = slab_case(tmp_path)
    defaults = case.settings()["design"]

    def transport(*, design_values=None):
        """Declare the two native material layers from one physical width.

        Parameters
        ----------
        design_values : dict or None, optional
            Caller-supplied physical design; None selects the configured initial design.

        Returns
        -------
        dict
            Design values and two equal-volume material regions.
        """
        values = defaults if design_values is None else design_values
        return {"design": values, "regions": [
            {"label": name, "volume_cm3": 5. * values["width_cm"][0]} for name in ("fuel", "moderator")]}

    def mesh(design, native):
        """Discretize the native two-layer geometry without homogenizing it.

        Parameters
        ----------
        design : SlabDesign
            Live physical slab width.
        native : dict
            Native region identities in physical layer order.

        Returns
        -------
        DiffusionGeometry
            Two Cartesian cells with one native material in each cell.
        """
        grid = Grid3D((tensor([0., .5, 1.]) * design.width_cm[0], [0., 2.], [0., 5.]))
        return DiffusionGeometry(grid, lambda: torch.eye(2, dtype=torch.float64),
                                 tuple(region["label"] for region in native["regions"]))

    def surrogate(design, state, layout, *, mesh):
        """Bind a supplied geometry to the generic material-calibration model.

        Parameters
        ----------
        design : SlabDesign
            Live design already bound by the geometry mapper.
        state : dict
            Native case used by the shared geometry-selection path.
        layout : str
            Unused calibration grouping for this zero-correction fixture.
        mesh : DiffusionGeometry
            Explicitly selected geometry; this factory does not choose another one.

        Returns
        -------
        CalibratedDiffusion
            Two-region diffusion model with unit material corrections.
        """
        absorption, fission = tensor([[.08], [.02]]), tensor([[.06], [0.]])
        scatter = tensor([[[.2]], [[.3]]])
        macro = {"absorption": absorption, "fission": fission, "nu_fission": 2.5 * fission,
                 "kappa_fission": 2e8 * fission, "scatter": scatter, "nu_scatter": scatter,
                 "transport": absorption + scatter.sum(-1), "chi": torch.ones_like(fission)}
        return CalibratedDiffusion(Diffusion3D(mesh.grid), mesh.fractions, macro, {}, [], power_W=100.)

    return replace(case, model=replace(case.model, transport=transport, mesh=mesh, surrogate=surrogate))


@pytest.mark.parametrize("simplified", [False, True])
def test_native_geometry_is_default_and_simplification_is_explicit(tmp_path, simplified):
    """Use the same declared geometry choice in preparation and calibration.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated input and output directory supplied by pytest.
    simplified : bool
        Explicitly select homogeneous mixing instead of the native material layers.
    """
    case = spatial_slab_case(tmp_path)
    native = case.make_case()
    original = json.dumps(native, sort_keys=True)
    original_definition = case.definition()
    if simplified:
        native_mapper = case.model.mesh

        def homogenized(design, native_case):
            """Replace layers with a uniform mixture while conserving both volumes.

            Parameters
            ----------
            design : SlabDesign
                Live physical slab dimensions.
            native_case : dict
                Unmodified native material-region definition.

            Returns
            -------
            DiffusionGeometry
                Explicitly simplified layout with the same material identities and volumes.
            """
            geometry = native_mapper(design, native_case)
            return replace(geometry, fractions=lambda: torch.full((2, 2), .5, dtype=torch.float64))

        case = replace(case, model=replace(case.model, surrogate_mesh=homogenized))
    prepared = prepare_case(case)
    metadata = json.loads((prepared / "resolved_inputs.json").read_text())
    assert metadata["surrogate_geometry"] == ("explicit" if simplified else "native")
    assert (case.definition() != original_definition) == simplified
    design = case.design()
    model = case.model.make_surrogate(design, {"case": native}, "unused")
    expected = torch.full((2, 2), .5, dtype=torch.float64) if simplified else torch.eye(2, dtype=torch.float64)
    torch.testing.assert_close(model.geometry(), expected)
    assert json.dumps(native, sort_keys=True) == original
    output = model()
    assert float(output["keff"].detach()) > 0
    torch.testing.assert_close(output["volume_cm3"], design.width_cm.expand(2) * 5.)


def test_simplified_geometry_cannot_silently_reorder_native_materials(tmp_path):
    """Reject an ambiguous material mapping even when all volumes still match.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated input and output directory supplied by pytest.
    """
    case = spatial_slab_case(tmp_path)
    mapper = case.model.mesh
    case = replace(case, model=replace(case.model, surrogate_mesh=lambda design, native:
        replace(mapper(design, native), region_labels=("moderator", "fuel"))))
    with pytest.raises(ValueError, match="preserve native region identities"):
        prepare_case(case)


def test_simplified_geometry_must_preserve_represented_material_volumes(tmp_path):
    """Reject a simplified layout that changes the amount of each material.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated input and output directory supplied by pytest.
    """
    case = spatial_slab_case(tmp_path)
    mapper = case.model.mesh
    case = replace(case, model=replace(case.model, surrogate_mesh=lambda design, native:
        replace(mapper(design, native), fractions=lambda: torch.tensor([[.75, .25]] * 2, dtype=torch.float64))))
    with pytest.raises(ArithmeticError, match="region volumes differ"):
        prepare_case(case)


def test_alternative_mapper_cannot_mutate_native_geometry(tmp_path):
    """Isolate any changes made while constructing an alternative surrogate layout.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated input and output directory supplied by pytest.
    """
    case = spatial_slab_case(tmp_path)
    mapper = case.model.mesh

    def alternative(design, definition):
        """Modify only the isolated definition supplied to this mapper.

        Parameters
        ----------
        design : SlabDesign
            Live design passed to the ordinary discretization callback.
        definition : dict
            Private copy of the native region definition.

        Returns
        -------
        DiffusionGeometry
            Mapping retaining the original physical material volumes.
        """
        definition["regions"][0]["volume_cm3"] *= 2.
        return mapper(design, definition)

    case = replace(case, model=replace(case.model, surrogate_mesh=alternative))
    native = case.make_case()
    original = json.dumps(native, sort_keys=True)
    case.model.make_mesh(case.design(), native)
    assert json.dumps(native, sort_keys=True) == original
    prepare_case(case)


class SlabDesign(torch.nn.Module):
    def __init__(self):
        """Create generic slab width, fuel-scale and cycle-period parameters."""
        super().__init__()
        self.width_cm = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
        self.fuel_scale = torch.nn.Parameter(torch.tensor([1.5], dtype=torch.float64))
        self.cycle_days = torch.nn.Parameter(torch.tensor([4.], dtype=torch.float64))


def slab_case(tmp_path):
    """Build a minimal case fixture without cylindrical reactor geometry.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.

    Returns
    -------
    CycleCase
        Slab workflow with isolated inputs, results and a text-only export callback.
    """
    settings = {"design": parameters(SlabDesign()), "cycles": 2, "objective_cycle": 2,
                "differentiated_cycles": [1, 2], "refueling": {"cycle_days_bounds": [1., 10.]}}
    options = {"bounds": {"width_cm": [1., 5.], "fuel_scale": [.5, 3.], "cycle_days": [1., 10.]},
               "trust_half_width": {"width_cm": .3, "fuel_scale": .1, "cycle_days": .5}}
    inputs = {}
    for kind, data in (("cycle", settings), ("optimization", options), ("calibration", {})):
        inputs[kind] = tmp_path / f"{kind}.json"
        inputs[kind].write_text(json.dumps(data))

    def transport(*, design_values=None):
        """Build a slab region from the supplied width parameter.

        Parameters
        ----------
        design_values : mapping or None, optional
            Explicit design values; None uses the fixture's configured design.

        Returns
        -------
        dict
            Design and one region with its volume in cm³.
        """
        values = settings["design"] if design_values is None else design_values
        return {"design": values, "regions": [{"volume_cm3": values["width_cm"][0] * 10.}]}

    def export(native, directory):
        """Write the slab volume through the generic native-export hook.

        Parameters
        ----------
        native : mapping
            Prepared slab case containing one region.
        directory : pathlib.Path
            Prepared-input directory receiving slab.txt.
        """
        (directory / "slab.txt").write_text(str(native["regions"][0]["volume_cm3"]))

    model = CaseModel(design=SlabDesign, transport=transport, inventory=lambda values: None,
                      surrogate=lambda *args, **kwargs: None, equivalence_bases=equivalence_bases,
                      fitting_targets=lambda *args: (), export_transport=export)
    return CycleCase(name="slab", model=model, inputs=inputs, work=tmp_path / "work",
                     results=tmp_path / "results", archive_root=tmp_path)


def test_preparation_and_projection_use_the_callers_parameters(tmp_path):
    """Verify preparation and projection use the callers parameters.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    case = slab_case(tmp_path)
    directory = prepare_case(case)
    assert (directory / "slab.txt").read_text() == "20.0"
    recorded = json.loads((directory / "resolved_inputs.json").read_text())
    assert recorded == case.definition()
    assert set(recorded["settings"]) == {"cycle", "calibration", "optimization"}
    design = case.design()
    anchor = parameters(design)
    set_parameters(design, {name: [100.] for name in anchor})
    project(design, anchor, case.settings("optimization"), case)
    torch.testing.assert_close(design.width_cm, torch.tensor([2.3], dtype=torch.float64))
    torch.testing.assert_close(design.fuel_scale, torch.tensor([1.6], dtype=torch.float64))
    torch.testing.assert_close(design.cycle_days, torch.tensor([4.5], dtype=torch.float64))


def test_preparation_rejects_a_factory_that_ignores_case_values(tmp_path):
    """Verify preparation rejects a factory that ignores case values.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    case = slab_case(tmp_path)
    settings = case.settings()
    settings["design"]["width_cm"] = [3.]
    case.input().write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="declared initial parameter width_cm"):
        prepare_case(case)


def test_interrupted_search_resumes_without_rebuilding_its_baseline(tmp_path, monkeypatch):
    """Verify interrupted search resumes without rebuilding its baseline.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacements for dependencies or environment settings.
    """
    from coupling import history, optimization

    case = slab_case(tmp_path)
    case.prepare_outputs()
    record = {"status": "interrupted", "case_definition": case.definition()}
    case.result("optimization").write_text(json.dumps(record))

    def unexpected(*args, **kwargs):
        """Fail if a resumed search tries to replace its existing baseline.

        Parameters
        ----------
        *args : object
            Positional arguments accepted only to detect an unexpected call.
        **kwargs : object
            Keyword arguments accepted only to detect an unexpected call.
        """
        pytest.fail("A resumed search must preserve the baseline associated with its checkpoint")

    for name in ("prepare_baseline", "check_sampling", "check_replay"):
        monkeypatch.setattr(history, name, unexpected)
    resumed = []
    monkeypatch.setattr(optimization, "run_search", lambda value: resumed.append(value))
    run_case(case)
    assert resumed == [case]
    assert json.loads(case.result("optimization").read_text()) == record


def test_unfinished_baseline_cannot_be_reused_for_different_inputs(tmp_path):
    """Reject a baseline with different physical input values.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    case = slab_case(tmp_path)
    case.prepare_outputs()
    case.result("history").write_text("{}")
    states = case.result("states", "")
    states.mkdir()
    definition = case.definition()
    definition["settings"]["cycle"]["design"]["width_cm"] = [3.]
    (states / "manifest.json").write_text(json.dumps({"case_definition": definition}))
    with pytest.raises(ValueError, match="native baseline uses different inputs"):
        run_case(case)


@pytest.mark.parametrize("status", ["search_stopped", "local_step_review_needed"])
def test_stopped_search_requires_a_new_run(tmp_path, monkeypatch, status):
    """Verify stopped search requires a new run.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacements for dependencies or environment settings.
    status : str
        Retained stopped-search status that must prevent automatic resumption.
    """
    from coupling import optimization

    case = slab_case(tmp_path)
    case.prepare_outputs()
    case.result("optimization").write_text(json.dumps({"status": status}))
    monkeypatch.setattr(optimization, "refresh_native", lambda *args: pytest.fail("Search is stopped"))
    optimization.run_search(case)
    assert json.loads(case.result("optimization").read_text()) == {"status": status}


def test_termination_unwinds_cleanup_and_restores_the_previous_signal_handler():
    """Verify termination unwinds cleanup and restores the previous signal handler."""
    previous = signal.getsignal(signal.SIGTERM)
    cleaned = []
    with pytest.raises(KeyboardInterrupt, match="signal"):
        with interrupt_on_sigterm():
            try:
                signal.raise_signal(signal.SIGTERM)
            finally:
                cleaned.append(True)
    assert cleaned == [True]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_two_level_residence_uses_a_custom_fresh_recipe_and_analytic_gradients():
    """Verify two level residence uses a custom fresh recipe and analytic gradients."""
    design = SlabDesign()
    labels = ["slab/axial/0", "slab/axial/1"]
    physical = SimpleNamespace(labels=labels, symmetry_multiplier=1,
        multiplicity=np.ones(2), regions=[{"label": label, "ring": 7} for label in labels],
        live_recipe=lambda d: ((2. * d.fuel_scale).expand(2, 1),
                              d.width_cm.expand(2), d.fuel_scale.expand(2)))
    mapping = FuelMap(physical)
    config = {"initial_inventory": "fresh", "objective_cycle": 2, "differentiated_cycles": [1, 2],
              "step_fractions": [0., 1.], "refueling": {"rings": [7]},
              "objective": {"reference_discharge_burnup_MWd_tHM": 5000.,
                            "assembly_power_weight": 0., "keff_deficit_scale": .01}}
    records, points = [], {}
    for cycle in (1, 2):
        bol, mid, end = (f"cycle{cycle}/{name}" for name in ("bol", "s0_mid", "s0_end"))
        records.append({"cycle": cycle, "bol": bol, "segments": [
            {"start": bol, "mid": mid, "end": end, "end_key": "step/0", "reload_rings": []}]})
        for key in (bol, mid, end):
            points[key] = SimpleNamespace(reference_boron=2., evaluate=lambda inventory, boron: {
                "rates": torch.full((2, 1, 1), .01 / DAY, dtype=torch.float64),
                "zone_power_w": torch.full((2,), 1e6, dtype=torch.float64),
                "keff": torch.tensor(1.1, dtype=torch.float64), "boron_ppm": boron})
    chain = lambda inventory, rates, seconds: inventory * torch.exp(-rates[..., 0] * seconds)
    result, state, _, _ = replay_history(mapping, chain, design, {"cycles": records}, points, config)
    expected_burnup = 2. * design.cycle_days * 1000. / design.fuel_scale
    torch.testing.assert_close(result["discharge_burnup_MWd_tHM"], expected_burnup.squeeze())
    torch.testing.assert_close(state.inventory[-1, 0],
        (2. * design.fuel_scale * torch.exp(-.02 * design.cycle_days)).squeeze())
    days, mass = torch.autograd.grad(result["discharge_burnup_MWd_tHM"],
                                    (design.cycle_days, design.fuel_scale))
    torch.testing.assert_close(days, 2000. / design.fuel_scale)
    torch.testing.assert_close(mass, -2000. * design.cycle_days / design.fuel_scale.square())


def test_equivalence_groups_and_isotope_aliases_are_explicit():
    """Verify equivalence groups and isotope aliases are explicit."""
    names, basis, equivalence = equivalence_bases(["left", None, "right", "left"], 3)
    assert len(names) == 9
    assert basis["diffusion"].shape == (9, 4, 3)
    assert not equivalence[:, 1].any()
    torch.testing.assert_close(equivalence[:, 0], equivalence[:, 3])
    response = {"nuclides": ["C12", "C13"],
                "microscopic": {"absorption": [[[1.], [10.]]]}}
    table = InventoryCrossSections(response, [{"inert_atom_density": {}}], ("C12", "C13"))
    result = table(torch.tensor([[2e24, 3e24]], dtype=torch.float64))["absorption"]
    torch.testing.assert_close(result, torch.tensor([[32.]], dtype=torch.float64))


def test_native_acceptance_uses_the_callers_bounds(tmp_path):
    """Verify native acceptance uses the callers bounds.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    from coupling.optimization import refresh_native_acceptance

    case = slab_case(tmp_path)
    options = case.settings("optimization")
    options.update(native_keff_standard_errors=2., native_acceptance=["declared bounds"])
    result = {"feasible": True, "metrics": {"discharge_burnup_MWd_tHM": 1000., "power_variance": .02},
              "design": parameters(case.design()), "merit": -.3}
    record = {"initial_native": result, "batches": []}
    refresh_native_acceptance(record, options)
    assert record["best_native"]["batch"] == 0
    assert record["native_acceptance"]["parameter_bounds"] == options["bounds"]
    result["design"]["width_cm"] = [100.]
    refresh_native_acceptance(record, options)
    assert record["best_native"] is None


def test_native_export_does_not_require_boron(tmp_path):
    """Verify native export does not require boron.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    from diffusion_surrogate.openmc_cylinders import build

    shape = {"xy_cm": [0., 0.], "inner_radius_cm": 0., "outer_radius_cm": .5}
    native = {
        "geometry": {"symmetry": "full", "apothem_cm": 2., "lower_cm": [-3., -3., 0.],
                     "upper_cm": [3., 3., 1.], "channel_shapes": [shape]},
        "sampling": {"particles": 100, "batches": 10, "inactive": 2, "seed": 9},
        "regions": [
            {"label": "fuel", "pieces": [shape], "z_bounds_cm": [0., 1.], "depletable": True,
             "void": False, "atom_density": {"U235": .02}, "temperature_K": 900., "volume_cm3": np.pi / 4},
            {"label": "coolant", "z_bounds_cm": [0., 1.], "depletable": False, "void": False,
             "atom_density": {"Na23": .02}, "temperature_K": 600., "volume_cm3": 8 * np.sqrt(3) - np.pi / 4}],
        "cross_sections": str(tmp_path / "cross_sections.xml"), "temperature_label_tolerance_K": 1.,
        "comparison_edges_cm": [[-3., 3.], [-3., 3.], [0., 1.]], "group_edges_eV": [0., 2e7]}
    model, materials = build(native, tmp_path)
    assert {tally.id for tally in model.tallies}.isdisjoint({22, 23, 24})
    assert [material.temperature for material in materials] == [900., 600.]
    assert all(not any(name.startswith("B") for name in material.get_nuclides()) for material in materials)
    assert (tmp_path / "geometry.xml").is_file()


def test_regional_surrogate_without_boron_rejects_an_unsupported_control_change():
    """Verify regional surrogate without boron rejects an unsupported control change."""
    from diffusion_surrogate.regional import RegionalSurrogate
    from diffusion_surrogate.solver3d import Diffusion3D, Grid3D

    grid = Grid3D(([0., 1.], [0., 1.], [0., 1.]))
    native = {"power_W": 1., "comparison_edges_cm": grid.edges, "group_edges_eV": [0., 2e7],
              "regions": [{"depletable": True}]}
    reference = {"macro": {"absorption": [[.1]]}, "region_labels": ["fuel"],
                 "material_response": {"microscopic": {"absorption": [[[1.]]]}}}
    model = RegionalSurrogate(SlabDesign(), native, reference,
        geometry=lambda: torch.ones((1, 1), dtype=torch.float64), solver=Diffusion3D(grid),
        names=[], bases={}, equivalence=None, fuel_volumes=lambda: grid.volumes)
    torch.testing.assert_close(model.boron_material_delta(0.)["absorption"], torch.zeros_like(model.macro_absorption))
    with pytest.raises(ValueError, match="no calibrated boron response"):
        model.boron_material_delta(1.)


def test_fresh_recipe_preserves_native_rounding_and_live_derivatives():
    """Verify fresh recipe preserves native rounding and live derivatives."""
    from materials.composition import UO2Recipe

    recipe = UO2Recipe(10., .7, {"O16": 1.})
    nuclides = ("U235", "U238", "O16")
    enrichment = torch.tensor([.025, .04], dtype=torch.float64, requires_grad=True)
    volumes = torch.tensor([1., 2.], dtype=torch.float64)
    direct, _, _ = recipe.live_inventory(enrichment, nuclides, volumes, {})
    native = torch.nextafter(direct.detach(), torch.full_like(direct, float("inf")))
    reference = (enrichment.detach().clone(), native)
    anchored, _, _ = recipe.live_inventory(enrichment, nuclides, volumes, {}, reference=reference)
    torch.testing.assert_close(anchored, native, rtol=0., atol=0.)
    torch.testing.assert_close(torch.autograd.grad(anchored.sum(), enrichment)[0],
                              torch.autograd.grad(direct.sum(), enrichment)[0])
    changed, _, _ = recipe.live_inventory(enrichment + .001, nuclides, volumes, {}, reference=reference)
    unanchored, _, _ = recipe.live_inventory(enrichment + .001, nuclides, volumes, {})
    torch.testing.assert_close(changed, native + (unanchored - direct), rtol=0., atol=0.)
