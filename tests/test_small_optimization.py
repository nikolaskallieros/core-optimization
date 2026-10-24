from fixtures.small_case import study as small_study
from examples.wr_1 import WR1_ROOT
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fuel_cycle.residence import FuelMap, FuelState, J_PER_MWD, metrics
from coupling.optimization import discharge_summary, design_loss, native_power_profile, prediction_rank, project, refresh_native_acceptance
from runtime.io import save_checkpoint

ROOT = WR1_ROOT
FIXTURES = Path(__file__).resolve().parent / "fixtures/wr1"
DTYPE = torch.float64


def config():
    """Read the small-cycle optimization fixture.

    Returns
    -------
    dict
        Retained optimization settings.
    """
    return json.loads((Path(__file__).parent / "fixtures/wr1/small_cycle_optimization.json").read_text())


def strict_config():
    """Select the strict criticality penalty for analytic checks.

    Returns
    -------
    dict
        Fresh fixture settings with surrogate_keff_policy set to strict.
    """
    # These retained analytic fixtures exercise the original strict penalty.
    options = config()
    options["surrogate_keff_policy"] = "strict"
    return options


def test_discharge_average_counts_all_rings_and_mass_once():
    """Verify discharge average counts all rings and mass once."""
    labels = [f"{c}/axial/{level}" for c in ("center", "lower", "upper") for level in range(3)]
    physical = SimpleNamespace(labels=labels, symmetry_multiplier=6,
        multiplicity=np.repeat([1, 3, 3], 3),
        regions=[{"label": label, "ring": int(not label.startswith("center"))} for label in labels])
    mapping = FuelMap(physical)
    # Different radii imply different assembly masses: the arithmetic average is wrong.
    mass = torch.repeat_interleave(torch.tensor([20., 40., 50.], dtype=DTYPE), 3) * mapping.fractions
    burnup = torch.tensor([1., 2., 5000., 1., 2., 7000., 1., 2., 9000.], dtype=DTYPE)
    state = FuelState(torch.ones((9, 1), dtype=DTYPE), burnup * mass / 1000 * J_PER_MWD, mass,
                      tuple(f"cycle1/{label}" for label in labels))
    summary = discharge_summary(mapping, state)
    expected = (20 * 5000 + 3 * 40 * 7000 + 3 * 50 * 9000) / (20 + 3 * 40 + 3 * 50)
    assert summary["physical_assemblies"] == 7
    assert summary["full_core_discharge_initial_hm_kg"] == pytest.approx(290.)
    assert summary["mean_burnup_MWd_tHM"] == pytest.approx(expected)
    assert expected != pytest.approx((5000 + 7000 + 9000) / 3)
    _, outgoing = mapping.shuffle(state, (0, 1), state.inventory, mass, "next")
    objective = {"reference_discharge_burnup_MWd_tHM": 7500., "assembly_power_weight": 1.,
                 "keff_deficit_scale": .01, "peak_burnup_MWd_tHM": 10800.}
    samples = [{"duration_days": torch.tensor(1.), "zone_power_w": mapping.fractions,
                "keff": torch.tensor(1.1), "peak_burnup": burnup.max()}]
    result = metrics(mapping, state, samples, [outgoing], {"objective": objective})
    assert float(result["discharge_burnup_MWd_tHM"]) == pytest.approx(expected)


def test_constraint_gradient_uses_limiting_midpoint_and_earlier_cycle_peak():
    """Verify constraint gradient uses limiting midpoint and earlier cycle peak."""
    objective = {"reference_discharge_burnup_MWd_tHM": 7500., "assembly_power_weight": 1.,
                 "peak_burnup_MWd_tHM": 10800.}
    values = {"discharge_burnup_MWd_tHM": torch.tensor(6000., dtype=DTYPE),
              "power_variance": torch.tensor(.08, dtype=DTYPE)}
    # Endpoints can be critical while a midpoint is subcritical.
    k = torch.tensor([1.02, .99, 1.03], dtype=DTYPE, requires_grad=True)
    peak = torch.tensor(11000., dtype=DTYPE, requires_grad=True)
    loss, score = design_loss(values, k, peak, objective, strict_config())
    kg, pg = torch.autograd.grad(loss, (k, peak))
    assert float(score) == pytest.approx(-6000. / 5000. + .1 * .08)
    assert kg[0] == kg[2] == 0 and kg[1] < 0 and pg > 0
    epsilon = 1e-6
    shifted = k.detach().clone()
    shifted[1] += epsilon
    plus = design_loss(values, shifted, peak.detach(), objective, strict_config())[0]
    shifted[1] -= 2 * epsilon
    minus = design_loss(values, shifted, peak.detach(), objective, strict_config())[0]
    assert float(kg[1]) == pytest.approx(float((plus - minus) / (2 * epsilon)), rel=1e-8)
    options = strict_config()
    variance = torch.tensor(.11, dtype=DTYPE, requires_grad=True)
    values["power_variance"] = variance
    loss = design_loss(values, k.detach(), peak.detach(), objective, options)[0]
    derivative, = torch.autograd.grad(loss, (variance,))
    epsilon = 1e-3  # Resolve the small linear power term beside the active k penalty.
    plus = design_loss({**values, "power_variance": variance.detach() + epsilon},
                      k.detach(), peak.detach(), objective, options)[0]
    minus = design_loss({**values, "power_variance": variance.detach() - epsilon},
                       k.detach(), peak.detach(), objective, options)[0]
    assert float(derivative) == .1
    assert float(derivative) == pytest.approx(float((plus - minus) / (2 * epsilon)), rel=1e-8)


def test_burnup_reward_continues_above_floor_with_lambda_point_one():
    """Verify burnup reward continues above floor with lambda point one."""
    options = strict_config()
    objective = {"peak_burnup_MWd_tHM": 10800.}
    k, peak = torch.tensor([1.02]), torch.tensor(9000.)
    variance = torch.tensor(.1, dtype=DTYPE)
    for value in (5000., 6000., 8000.):
        burnup = torch.tensor(value, dtype=DTYPE, requires_grad=True)
        loss, score = design_loss({"discharge_burnup_MWd_tHM": burnup, "power_variance": variance},
                                  k, peak, objective, options)
        assert float(loss) == float(score) == pytest.approx(-value / 5000. + .1 * .1)
        assert float(torch.autograd.grad(loss, burnup)[0]) == pytest.approx(-1. / 5000.)
    burnup = torch.tensor(4950., dtype=DTYPE, requires_grad=True)
    def evaluate(b):
        """Evaluate loss while varying only discharge burnup.

        Parameters
        ----------
        b : torch.Tensor
            Live discharge burnup in MWd/tHM.

        Returns
        -------
        torch.Tensor
            Penalized design objective at the fixed fixture power and criticality.
        """
        return design_loss({"discharge_burnup_MWd_tHM": b, "power_variance": variance},
                           k, peak, objective, options)[0]
    gradient, = torch.autograd.grad(evaluate(burnup), burnup)
    epsilon = .01
    finite = (evaluate(burnup.detach() + epsilon) - evaluate(burnup.detach() - epsilon)) / (2 * epsilon)
    assert float(gradient) < 0
    assert float(gradient) == pytest.approx(float(finite), rel=1e-8)


def test_surrogate_selection_cannot_trade_the_burnup_constraint_for_flatter_power():
    """Verify surrogate selection cannot trade the burnup constraint for flatter power."""
    options = config()
    options["surrogate_keff_policy"] = "strict"
    anchor = {"objective": .12, "metrics": {"discharge_burnup_MWd_tHM": 5700.},
              "minimum_keff": 1.03, "peak_constraint_passed": True}
    infeasible = {**anchor, "objective": .08, "metrics": {"discharge_burnup_MWd_tHM": 4999.}}
    feasible = {**infeasible, "metrics": {"discharge_burnup_MWd_tHM": 5000.}}
    assert prediction_rank(feasible, options) < prediction_rank(anchor, options)
    assert prediction_rank(anchor, options) < prediction_rank(infeasible, options)
    assert prediction_rank(anchor, options) < prediction_rank({**feasible, "minimum_keff": .99}, options)
    assert prediction_rank(anchor, options) < prediction_rank({**feasible, "peak_constraint_passed": False}, options)


def test_native_power_profile_preserves_time_variation_and_physical_multiplicity():
    """Verify native power profile preserves time variation and physical multiplicity."""
    labels = [f"{c}/axial/{level}" for c in ("center", "lower", "upper") for level in range(3)]
    physical = SimpleNamespace(labels=labels, symmetry_multiplier=6,
        multiplicity=np.repeat([1, 3, 3], 3),
        regions=[{"label": label, "ring": int(not label.startswith("center"))} for label in labels])
    mapping = FuelMap(physical)
    variation = np.tile([20000., -40000., 20000.], 3)
    observations = {f"m{i}": {"power_W": ((100000. + sign * variation) * mapping.fractions.numpy()).tolist()}
                    for i, sign in enumerate([1., -1.])}
    history = {"observations": observations, "cycles": [{"cycle_days": 2.,
        "segments": [{"mid": "m0", "duration_days": 1.}, {"mid": "m1", "duration_days": 1.}],
        "metrics": {"power_variance": .08}}]}
    profile = native_power_profile(history, mapping)
    assert sum(profile["physical_multiplicity"]) == 21
    assert profile["full_core_power_W"] == pytest.approx(2.1e6)
    np.testing.assert_allclose(profile["mean_power_W"], 100000.)
    # A flat time-average profile must not erase nonuniform instantaneous power.
    assert profile["power_variance_recomputed"] == pytest.approx(.08)
    assert profile["channel_power_variance_recomputed"] == pytest.approx(0., abs=1e-25)
    np.testing.assert_allclose(profile["channel_mean_power_W"], 300000.)
    assert profile["channel_physical_multiplicity"] == [1, 3, 3]


def test_native_selection_minimizes_power_variance_subject_to_burnup_and_bounds():
    """Verify native selection minimizes power variance subject to burnup and bounds."""
    options = config()
    options["objective_kind"] = "power_flatness"  # Retained earlier search semantics.
    design = {"pitch_mm": [240.], "element_radius_mm": [38., 38.],
              "enrichment": [.031, .038], "cycle_days": [110.]}
    initial = {"feasible": True, "design": deepcopy(design), "merit": -.4,
               "metrics": {"power_variance": .10, "discharge_burnup_MWd_tHM": 3597.}}
    poor_power = {"feasible": True, "design": deepcopy(design), "merit": -.9,
                  "metrics": {"power_variance": .12, "discharge_burnup_MWd_tHM": 5769.}}
    record = {"initial_native": initial, "batches": [{"number": 1, "native": poor_power}]}
    refresh_native_acceptance(record, options)
    assert record["best_native"]["batch"] == 1
    assert not initial["acceptable"] and not initial["burnup_passed"]
    good = {"feasible": True, "design": deepcopy(design), "merit": -.8,
            "metrics": {"power_variance": .11, "discharge_burnup_MWd_tHM": 5000.}}
    record["batches"].append({"number": 2, "native": good})
    refresh_native_acceptance(record, options)
    assert record["best_native"]["batch"] == 2
    assert record["best_native"]["selection_merit"] == .11
    good["design"]["enrichment"][0] = .0069
    refresh_native_acceptance(record, options)
    assert not good["bounds_passed"] and record["best_native"]["batch"] == 1
    good["design"]["enrichment"] = [.007, .05]
    refresh_native_acceptance(record, options)
    assert good["acceptable"] and record["best_native"]["batch"] == 2
    good["physics_passed"] = False
    refresh_native_acceptance(record, options)
    assert not good["acceptable"] and record["best_native"]["batch"] == 1


def test_projection_preserves_fixed_power_height_and_valid_geometry():
    """Verify projection preserves fixed power height and valid geometry."""
    from fixtures.small_case import SmallDesign, make_case, settings
    from optimization.design import parameters
    design = SmallDesign(ROOT)
    anchor = parameters(design)
    settings_before = settings(ROOT)
    with torch.no_grad():
        design.pitch_mm.fill_(1.)
        design.element_radius_mm.fill_(1000.)
        design.enrichment.fill_(1.)
        design.cycle_days.fill_(1000.)
    project(design, anchor, config(), small_study(ROOT))
    case = make_case(ROOT, design_values=parameters(design))
    assert case["power_W"] == settings_before["power_W"]
    assert float(design.z_edges_mm[-1]) == 2970.
    for name, value in parameters(design).items():
        lower, upper = config()["bounds"][name]
        assert all(lower <= x <= upper for x in value)
        assert max(abs(np.array(value) - anchor[name])) <= config()["trust_half_width"][name]
    assert design.enrichment.requires_grad and design.cycle_days.requires_grad


def test_adam_checkpoint_resumes_the_same_next_step(tmp_path):
    """Verify adam checkpoint resumes the same next step.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    from optimization.solvers import minimize
    design = torch.nn.ParameterDict({"x": torch.nn.Parameter(torch.tensor([2., -1.], dtype=DTYPE))})
    optimizer = torch.optim.Adam(design.parameters(), lr=.1)
    objective = lambda: ((design["x"] - torch.tensor([.5, .75], dtype=DTYPE)) ** 2).sum()
    minimize(design, objective, lambda value: value, optimizer=optimizer, steps=2)
    checkpoint = {"design": deepcopy(design.state_dict()), "optimizer": deepcopy(optimizer.state_dict()),
                  "batch": {"number": 1, "completed_steps": 2}}
    path = tmp_path / "optimizer.pt"
    save_checkpoint(path, checkpoint)
    minimize(design, objective, lambda value: value, optimizer=optimizer, steps=1)
    expected = design["x"].detach().clone()
    restored = torch.load(path, weights_only=True)
    design.load_state_dict(restored["design"])
    optimizer.load_state_dict(restored["optimizer"])
    minimize(design, objective, lambda value: value, optimizer=optimizer, steps=1)
    torch.testing.assert_close(design["x"], expected, rtol=0., atol=0.)
    assert restored["batch"]["completed_steps"] == 2


@pytest.mark.parametrize("enrichment, expected", [(-1., .007), (1., .05)])
def test_enrichment_projection_enforces_the_user_bounds(enrichment, expected):
    """Verify enrichment projection enforces the user bounds.

    Parameters
    ----------
    enrichment : float
        Initial U-235 uranium mass fraction outside the allowed interval.
    expected : float
        Expected bounded U-235 uranium mass fraction after projection.
    """
    from fixtures.small_case import SmallDesign
    from optimization.design import parameters
    design = SmallDesign(ROOT)
    options = config()
    options["trust_half_width"]["enrichment"] = 1.
    anchor = parameters(design)
    with torch.no_grad():
        design.enrichment.fill_(enrichment)
    project(design, anchor, options, small_study(ROOT))
    torch.testing.assert_close(design.enrichment, torch.full_like(design.enrichment, expected), rtol=0., atol=0.)


def test_native_acceptance_checks_midpoints_and_all_three_cycles():
    """Verify native acceptance checks midpoints and all three cycles."""
    from fixtures.small_case import settings
    from coupling.optimization import native_summary
    from runtime.io import load_state
    history = load_state(FIXTURES / "small_cycle_history.json")
    fresh = load_state(FIXTURES / "small_cycle_states/fresh.json")
    regions = [r for r in fresh["case"]["regions"] if r["depletable"]]
    physical = SimpleNamespace(regions=regions, labels=[r["label"] for r in regions], symmetry_multiplier=6,
                               multiplicity=np.array([r["physical_multiplicity"] for r in regions]))
    mapping = FuelMap(physical)
    result = native_summary(history, mapping, settings(ROOT), config())
    assert len(result["keff_by_state"]) == 33
    assert result["discharge"]["physical_assemblies"] == 7
    assert result["burnup_passed"] and result["bounds_passed"]
    assert result["merit"] == pytest.approx(-result["metrics"]["discharge_burnup_MWd_tHM"] / 5000.
                                          + .1 * result["metrics"]["power_variance"])
    midpoint = history["cycles"][0]["segments"][0]["mid"]
    history["observations"][midpoint]["keff"] = [.99, .001]
    result = native_summary(history, mapping, settings(ROOT), config())
    assert not result["feasible"]
    assert result["minimum_keff_minus_2se"] <= .988
    history["observations"][midpoint]["keff"] = [1.0019, .001]
    assert not native_summary(history, mapping, settings(ROOT), config())["feasible"]
    history["observations"][midpoint]["keff"] = [1.0021, .001]
    checked = native_summary(history, mapping, settings(ROOT), config())
    assert checked["feasible"]  # Nominal exploration; uncertainty remains explicit.
    assert checked["minimum_keff_minus_2se"] < 1.002
    history["cycles"][0]["segments"][0]["burnup_MWd_tHM"][0] = 11000.
    result = native_summary(history, mapping, settings(ROOT), config())
    assert result["maximum_peak_burnup_MWd_tHM"] >= 11000.


def test_channel_objective_ignores_axial_shape_and_has_correct_radial_derivative():
    """Verify channel objective ignores axial shape and has correct radial derivative."""
    from fixtures.small_case import settings
    labels = [f"{c}/axial/{z}" for c in ("center", "lower", "upper") for z in range(3)]
    mapping = FuelMap(SimpleNamespace(labels=labels, symmetry_multiplier=6,
        multiplicity=np.repeat([1, 3, 3], 3),
        regions=[{"label": s, "ring": int(not s.startswith("center"))} for s in labels]))
    config_values = settings(ROOT)
    outgoing = [{"energy_j": torch.tensor([6000. * J_PER_MWD], dtype=DTYPE),
                 "initial_hm_kg": torch.tensor([1000.], dtype=DTYPE)}]
    def evaluate(channel_totals, shapes):
        """Evaluate channel metrics for a specified axial power split.

        Parameters
        ----------
        channel_totals : torch.Tensor
            Complete physical channel powers in W.
        shapes : torch.Tensor
            Normalized per-channel axial power fractions.

        Returns
        -------
        dict
            Fixture cycle objective metrics using reconstructed represented-region powers.
        """
        power = (channel_totals[:, None] * shapes).reshape(-1) * mapping.fractions
        samples = [{"duration_days": torch.tensor(1.), "zone_power_w": power,
                    "keff": torch.tensor(1.02), "peak_burnup": torch.tensor(6000.)}]
        return metrics(mapping, None, samples, outgoing, config_values)
    shapes = torch.tensor([[.1, .7, .2], [.4, .4, .2], [.3, .1, .6]], dtype=DTYPE)
    equal = evaluate(torch.full((3,), 300000., dtype=DTYPE), shapes)
    assert float(equal["power_variance"]) == pytest.approx(0., abs=1e-25)
    assert float(equal["assembly_power_variance"]) > .1
    totals = torch.tensor([300000., 600000., 900000.], dtype=DTYPE, requires_grad=True)
    result = evaluate(totals, shapes)
    weights = torch.tensor([1., 3., 3.], dtype=DTYPE) / 7
    expected = (weights * (totals / (weights * totals).sum() - 1).square()).sum()
    torch.testing.assert_close(result["power_variance"], expected)
    changed_shape = evaluate(totals, torch.full((3, 3), 1/3, dtype=DTYPE))
    torch.testing.assert_close(changed_shape["power_variance"], result["power_variance"])
    derivative, = torch.autograd.grad(result["loss"], totals)
    delta = torch.tensor([1., -2., 3.], dtype=DTYPE)
    finite = (evaluate(totals.detach() + delta, shapes)["loss"]
              - evaluate(totals.detach() - delta, shapes)["loss"]) / 2
    torch.testing.assert_close(derivative @ delta, finite, rtol=1e-7, atol=1e-12)


def test_weighted_score_prefers_burnup_and_respects_the_new_k_limit():
    """Verify weighted score prefers burnup and respects the new k limit."""
    from coupling.optimization import merit
    options = config()
    options["surrogate_keff_policy"] = "strict"
    assert merit(6000., .1, {}, options) < merit(5000., 0., {}, options)
    assert merit(6000., .01, {}, options) - merit(6000., 0., {}, options) == pytest.approx(5. / 5000.)
    candidate = {"objective": -1.2, "metrics": {"discharge_burnup_MWd_tHM": 6000.},
                 "minimum_keff": 1.002, "peak_constraint_passed": True}
    assert prediction_rank(candidate, options)[0] == 0
    assert prediction_rank({**candidate, "minimum_keff": 1.0019}, options)[0] == 1


def test_fixed_fit_batch_recovers_after_two_steps_without_repeating_them(tmp_path, monkeypatch):
    """Verify fixed fit batch recovers after two steps without repeating them.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacements for dependencies or environment settings.
    """
    import coupling.optimization as optimization
    from fixtures.small_case import SmallDesign
    options = config()
    options["surrogate_keff_policy"] = "strict"

    class ToyObjective(torch.nn.Module):
        def __init__(self, shared, config):
            """Bind a manufactured objective to a restartable design fixture.

            Parameters
            ----------
            shared : types.SimpleNamespace
                Design and reference values used by the fixture.
            config : mapping
                Optimization settings accepted by the replacement objective interface.
            """
            super().__init__()
            self.design = shared.design
            self.coefficient = torch.nn.Parameter(torch.tensor(2., dtype=DTYPE))
            self.reference = shared.reference
            self.latest = None

        def forward(self):
            """Evaluate a scaled quadratic while retaining selection diagnostics.

            Returns
            -------
            dict of str to torch.Tensor
                Manufactured scalar loss for testing exact batch continuation.
            """
            loss = sum((((p - torch.tensor(self.reference[name], dtype=DTYPE))
                         / options["learning_rates"][name] - 3.5) ** 2).sum()
                       for name, p in self.design.named_parameters()) * self.coefficient
            self.latest = {"objective": float(loss.detach()), "minimum_keff": 1.1,
                           "peak_constraint_passed": True,
                           "metrics": {"loss": float(loss.detach()), "discharge_burnup_MWd_tHM": 6000.}}
            return {"loss": loss}

    monkeypatch.setattr(optimization, "CycleObjective", ToyObjective)

    def setup():
        """Create independent fresh optimizer-batch fixture objects.

        Returns
        -------
        tuple
            SmallDesign, shared namespace, batch record and checkpoint dictionary.
        """
        design = SmallDesign(ROOT)
        anchor = optimization.parameters(design)
        batch = {"anchor_design": anchor, "steps": []}
        checkpoint = {"phase": "surrogate", "batch": batch}
        shared = SimpleNamespace(design=design, reference=anchor)
        return design, shared, batch, checkpoint

    expected_design, shared, batch, checkpoint = setup()
    optimization.optimize_batch(small_study(ROOT), shared, expected_design, options, batch, checkpoint,
                                tmp_path / "expected.pt", lambda **kw: None, lambda: None)
    design, shared, batch, checkpoint = setup()
    path = tmp_path / "resumed.pt"

    def interrupted_save():
        """Interrupt deliberately after the second completed checkpoint.

        Raises
        ------
        KeyboardInterrupt
            Exactly two steps have been saved.
        """
        if checkpoint.get("completed_steps") == 2:
            raise KeyboardInterrupt("intentional interruption after a saved step")

    with pytest.raises(KeyboardInterrupt):
        optimization.optimize_batch(small_study(ROOT), shared, design, options, batch, checkpoint,
                                    path, lambda **kw: None, interrupted_save)
    recovered = torch.load(path, weights_only=True)
    assert recovered["completed_steps"] == len(recovered["batch"]["steps"]) == 2
    optimization.optimize_batch(small_study(ROOT), shared, design, options, recovered["batch"], recovered,
                                path, lambda **kw: None, lambda: None)
    assert [row["step"] for row in recovered["batch"]["steps"]] == [1, 2, 3, 4, 5]
    assert recovered["phase"] == "native"
    for name, p in design.named_parameters():
        torch.testing.assert_close(p, dict(expected_design.named_parameters())[name], rtol=0., atol=0.)
        assert not torch.equal(p, torch.tensor(shared.reference[name], dtype=DTYPE))
