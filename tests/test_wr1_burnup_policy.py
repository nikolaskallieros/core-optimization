"""WR-1 search bounds and uncapped burnup; no finite differences."""
from designs.wr_1 import WR1_ROOT
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fuel_cycle.residence import FuelMap, J_PER_MWD, metrics
from designs.wr_1.model.case import study
from coupling.optimization import design_loss, native_summary, prediction_rank, rescore_prediction
from runtime.io import load_state

ROOT = WR1_ROOT


def settings():
    """Load the default WR-1 cycle and optimization policy.

    Returns
    -------
    tuple of dict
        Cycle settings followed by optimization settings.
    """
    workflow = study()
    return (workflow.settings(), workflow.settings("optimization"))


@pytest.mark.parametrize("days", [209., 400.])
def test_native_period_range_matches_optimizer_above_the_old_limit(monkeypatch, tmp_path, days):
    """Verify native period range matches optimizer above the old limit.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped replacements for dependencies or environment settings.
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    days : float
        Cycle period in days above the removed historical limit.
    """
    from designs.wr_1.model.case import study
    from coupling.history import native_history

    case, options = settings()
    assert case["refueling"]["cycle_days_bounds"] == options["bounds"]["cycle_days"]
    workflow = study()

    class ReachedInventory(Exception):
        pass

    def inventory(self, values):
        """Verify the requested cycle period reaches the inventory factory.

        Parameters
        ----------
        values : mapping
            Design values forwarded by the native-history initializer.

        Raises
        ------
        ReachedInventory
            The expected period reaches the factory, ending the test before native transport.
        """
        assert values["cycle_days"] == [days]
        raise ReachedInventory

    from coupling.cases import CycleCase
    monkeypatch.setattr(CycleCase, "inventory", inventory)
    values = {**case["design"], "cycle_days": [days]}
    with pytest.raises(ReachedInventory):
        native_history(workflow, tmp_path, case, lambda **changes: None,
                       design_values=values, initialize=False)
    values["cycle_days"] = [options["bounds"]["cycle_days"][1] + 1.]
    with pytest.raises(ValueError, match="period exceeds"):
        native_history(workflow, tmp_path, case, lambda **changes: None,
                       design_values=values, initialize=False)


@pytest.mark.parametrize("burnup_value", [12000., 30000.])
def test_cycle_and_search_reward_burnup_above_the_removed_limit(burnup_value):
    """Verify cycle and search reward burnup above the removed limit.

    Parameters
    ----------
    burnup_value : float
        Discharge burnup used to check continued reward above the removed cap.
    """
    case, options = settings()
    assert case["objective"].get("peak_burnup_MWd_tHM") is None
    assert "surrogate_peak_penalty_scale" not in options
    labels = [f"center/axial/{z}" for z in range(3)]
    mapping = FuelMap(SimpleNamespace(labels=labels, symmetry_multiplier=6,
        multiplicity=np.ones(3), regions=[{"label": label, "ring": 0} for label in labels]))
    burnup = torch.tensor(burnup_value, dtype=torch.float64, requires_grad=True)
    peak = torch.tensor(2 * burnup_value, dtype=torch.float64, requires_grad=True)
    outgoing = [{"energy_j": (burnup * J_PER_MWD).reshape(1),
                 "initial_hm_kg": torch.tensor([1000.], dtype=torch.float64)}]
    samples = [{"duration_days": torch.tensor(1.), "zone_power_w": mapping.fractions,
                "keff": torch.tensor(1.1), "peak_burnup": peak}]
    result = metrics(mapping, None, samples, outgoing, case)
    loss, score = design_loss(result, torch.tensor([1.1]), peak, case["objective"], options)
    for value in (result["loss"], loss, score):
        assert float(value.detach()) == pytest.approx(-burnup_value / 5000.)
    gradient, peak_gradient = torch.autograd.grad(loss, (burnup, peak), allow_unused=True)
    assert float(gradient) == pytest.approx(-1. / 5000.)
    assert peak_gradient is None
    assert float(result["peak_burnup_MWd_tHM"].detach()) == 2 * burnup_value


def test_rescoring_removes_old_peak_rejection_but_preserves_the_burnup_minimum():
    """Verify rescoring removes old peak rejection but preserves the burnup minimum."""
    case, options = settings()
    original = {"objective": 100., "metrics": {"discharge_burnup_MWd_tHM": 20000.,
                 "power_variance": .01}, "minimum_keff": 1.04, "keff_by_state": {"eol": 1.04},
                "maximum_peak_burnup_MWd_tHM": 30000., "peak_constraint_passed": False}
    retained = deepcopy(original)
    scored = rescore_prediction(original, case["objective"], options, None)
    assert original == retained
    assert scored["peak_constraint_passed"] and scored["peak_burnup_limit_MWd_tHM"] is None
    assert prediction_rank(scored, options)[0] == 0
    assert scored["objective"] == pytest.approx(-4. + .001)
    original["metrics"]["discharge_burnup_MWd_tHM"] = 4999.
    assert prediction_rank(rescore_prediction(original, case["objective"], options, None), options)[0] == 1


def test_native_acceptance_has_no_peak_limit_and_still_checks_criticality():
    """Verify native acceptance has no peak limit and still checks criticality."""
    case, options = settings()
    workflow = study()
    if not workflow.result("history").is_file():
        pytest.skip("Optional regression requires locally generated WR-1 step-one results")
    history = load_state(workflow.result("history"))
    first = load_state(workflow.native_states() / f"{history['cycles'][0]['bol']}.json")
    regions = [region for region in first["case"]["regions"] if region["depletable"]]
    mapping = FuelMap(SimpleNamespace(regions=regions, labels=[r["label"] for r in regions],
        symmetry_multiplier=6, multiplicity=np.array([r["physical_multiplicity"] for r in regions])))
    for observation in history["observations"].values():
        observation["keff"] = [1.04, .003]
        observation["power_relative_se_rms"] = .02
        observation["power_relative_se_max"] = .04
    history["cycles"][0]["segments"][0]["burnup_MWd_tHM"][0] = 50000.
    result = native_summary(history, mapping, case, options)
    assert result["physics_passed"] and result["peak_constraint_passed"]
    assert result["maximum_peak_burnup_MWd_tHM"] >= 50000.
    limited = deepcopy(case)
    limited["objective"]["peak_burnup_MWd_tHM"] = 10800.
    assert not native_summary(history, mapping, limited, options)["physics_passed"]
    midpoint = history["cycles"][0]["segments"][0]["mid"]
    history["observations"][midpoint]["keff"] = [1.0019, .003]
    assert not native_summary(history, mapping, case, options)["physics_passed"]
