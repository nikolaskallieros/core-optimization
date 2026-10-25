"""Constraint-policy checks with analytic fixtures; no finite differences."""
from designs.wr_1 import WR1_ROOT
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from coupling.optimization import (corrected_keff, design_loss, keff_correction,
                                             prediction_rank, rescore_prediction)

ROOT = WR1_ROOT
DTYPE = torch.float64


def options():
    """Read the small-cycle criticality-selection policy fixture.

    Returns
    -------
    dict
        Retained small-cycle optimization options.
    """
    return json.loads((Path(__file__).parent / "fixtures/wr1/small_cycle_optimization.json").read_text())


def fixture_prediction(burnup, k):
    """Build explicit surrogate metrics for criticality-policy checks.

    Parameters
    ----------
    burnup : float
        Prescribed discharge burnup in MWd/tHM.
    k : float
        Prescribed limiting midpoint effective multiplication factor.

    Returns
    -------
    dict
        Synthetic prediction independent of any transport solve.
    """
    return {"objective": 0., "merit": 0., "minimum_keff": k,
            "keff_by_state": {"bol": 1.04, "mid": k, "eol": 1.01},
            "maximum_peak_burnup_MWd_tHM": 9000., "peak_constraint_passed": True,
            "metrics": {"loss": 0., "discharge_burnup_MWd_tHM": burnup,
                        "power_variance": .001}}


def fixed_correction():
    """Supply deterministic stage offsets and heuristic error scales.

    Returns
    -------
    dict
        Fixed BOL, midpoint and EOL correction data.
    """
    return {"offset_by_state": {k: .001 for k in ("bol", "mid", "eol")},
            "error_scale_by_state": {k: .005 for k in ("bol", "mid", "eol")}}


def test_anchor_matches_native_at_each_state_and_preserves_autograd():
    """Verify anchor matches native at each state and preserves autograd."""
    native = {"bol": {"keff": [1.05, .003]}, "eol": {"keff": [1.005, .004]}}
    surrogate = {"bol": {"keff": 1.04}, "eol": {"keff": 1.007}}
    correction = keff_correction(native, surrogate, options())
    raw = torch.tensor([1.04, 1.007], dtype=DTYPE, requires_grad=True)
    corrected, scale = corrected_keff(raw, ["bol", "eol"], correction)
    torch.testing.assert_close(corrected, torch.tensor([1.05, 1.005], dtype=DTYPE))
    torch.testing.assert_close(torch.autograd.grad(corrected.sum(), raw)[0], torch.ones_like(raw))
    assert torch.all(scale >= torch.tensor([.003, .004], dtype=DTYPE))


def test_missing_state_is_rejected():
    """Verify missing state is rejected."""
    with pytest.raises(ValueError, match="every matched"):
        keff_correction({"bol": {"keff": [1.1, .003]}}, {}, options())


def test_small_partial_violation_can_beat_lower_burnup_but_large_violation_cannot():
    """Verify small partial violation can beat lower burnup but large violation cannot."""
    config, objective = options(), {"peak_burnup_MWd_tHM": 10800.}
    score = lambda p: rescore_prediction(p, objective, config, fixed_correction())
    feasible = score(fixture_prediction(6000., 1.005))
    small = score(fixture_prediction(6500., .998))  # corrected .999; a .003 deficit.
    large = score(fixture_prediction(6500., .98))
    assert small["minimum_corrected_keff"] < config["surrogate_keff_target"]
    assert prediction_rank(small, config) < prediction_rank(feasible, config)
    assert prediction_rank(feasible, config) < prediction_rank(large, config)


def test_midpoint_penalty_has_the_expected_analytic_gradient():
    """Verify midpoint penalty has the expected analytic gradient."""
    config = options()
    raw = torch.tensor([1.02, .999, 1.01], dtype=DTYPE, requires_grad=True)
    scale = torch.full_like(raw, .005)
    values = {"discharge_burnup_MWd_tHM": torch.tensor(6000., dtype=DTYPE),
              "power_variance": torch.tensor(.001, dtype=DTYPE)}
    loss, _ = design_loss(values, raw, torch.tensor(9000.),
                         {"peak_burnup_MWd_tHM": 10800.}, config, scale)
    grad, = torch.autograd.grad(loss, raw)
    expected = -2 * config["surrogate_keff_soft_weight"] * .003 / .005**2
    assert grad.tolist() == pytest.approx([0., expected, 0.])


def test_rescoring_retains_original_predictions_and_burnup_guard():
    """Verify rescoring retains original predictions and burnup guard."""
    original = fixture_prediction(6500., .998)
    retained = deepcopy(original)
    config, objective = options(), {"peak_burnup_MWd_tHM": 10800.}
    scored = rescore_prediction(original, objective, config, fixed_correction())
    assert original == retained
    assert scored["keff_by_state"] == original["keff_by_state"]
    assert scored["minimum_keff"] == original["minimum_keff"]
    low_burnup = rescore_prediction(fixture_prediction(4999., 1.01), objective, config, fixed_correction())
    assert prediction_rank(scored, config) < prediction_rank(low_burnup, config)
    assert prediction_rank({**scored, "peak_constraint_passed": False}, config)[0] == 1
