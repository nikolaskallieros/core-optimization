"""Independent references for constrained searches and native acceptance."""

import numpy as np
import pytest
import torch
from scipy.optimize import OptimizeResult

from optimization.constrained import constrained_minimize
from optimization.design import Snapshot, optimize


class Quadratic(torch.nn.Module):
    """A known quadratic whose scale is a fitted, frozen model coefficient."""

    def __init__(self, design):
        """Bind physical coordinates and one independent scale.

        Parameters
        ----------
        design : torch.nn.ParameterDict
            Live coordinates owned by the design search.
        """
        super().__init__()
        object.__setattr__(self, "design", design)
        self.scale = torch.nn.Parameter(torch.tensor(1.3, dtype=torch.float64))
        self.calls = 0

    def forward(self):
        """Evaluate a scalar objective and two physical constraint quantities.

        Returns
        -------
        dict
            Quadratic loss and equality/inequality coordinates.
        """
        self.calls += 1
        values = self.design["x"]
        target = values.new_tensor([.9, .1, .5])
        return {"loss": self.scale * (values - target).square().sum(),
                "constraints": torch.stack((values.sum(), values[0]))}


def test_constrained_search_recovers_known_active_constraint_solution():
    """Recover an analytical optimum with a fixed sum and an active upper bound."""
    design = torch.nn.ParameterDict({"x": torch.nn.Parameter(torch.full((3,), .5, dtype=torch.float64))})
    model = Quadratic(design)
    result = constrained_minimize(design, model, lambda output: output["loss"],
        bounds=(0., 1.), constraints=lambda output: output["constraints"],
        constraint_bounds=([1.5, -np.inf], [1.5, .6]), steps=30, ftol=1e-12)
    torch.testing.assert_close(design["x"], torch.tensor([.6, .25, .65], dtype=torch.float64), rtol=0, atol=1e-9)
    np.testing.assert_allclose(result["gradient"]["x"], 1.3 * np.array([-.8, .8, 0.]), atol=1e-14)
    assert model.calls == result["function_evaluations"]
    assert float(model.scale.detach()) == 1.3
    assert model.scale.requires_grad


def test_failed_constrained_search_restores_the_starting_design():
    """Preserve a valid starting design when the declared constraints conflict."""
    design = torch.nn.ParameterDict({"x": torch.nn.Parameter(torch.full((3,), .5, dtype=torch.float64))})
    with pytest.raises(RuntimeError, match="Constrained design search failed"):
        constrained_minimize(design, Quadratic(design), lambda output: output["loss"],
            bounds=(0., 1.), constraints=lambda output: output["constraints"],
            constraint_bounds=([4., -np.inf], [4., .6]), steps=5, ftol=1e-12)
    torch.testing.assert_close(design["x"], torch.full((3,), .5, dtype=torch.float64), rtol=0, atol=0)


def test_two_fixed_response_equalities_recover_the_analytical_projection():
    """Keep two independent quantities fixed while improving a third response."""
    design = torch.nn.ParameterDict({"x": torch.nn.Parameter(torch.full((3,), .5, dtype=torch.float64))})
    model = Quadratic(design)
    weights = torch.tensor([1., 2., 3.], dtype=torch.float64)
    result = constrained_minimize(design, model, lambda output: output["loss"],
        bounds=(0., 1.), constraints=lambda output: torch.stack((design["x"].sum(), design["x"] @ weights)),
        constraint_bounds=([1.5, 3.], [1.5, 3.]), steps=30, ftol=1e-12)
    torch.testing.assert_close(design["x"], torch.tensor([.7, .1, .7], dtype=torch.float64), rtol=0., atol=1e-9)
    np.testing.assert_allclose(result["constraints"], [1.5, 3.], rtol=0., atol=1e-12)


def test_budget_limited_candidate_requires_feasibility_and_reports_nonconvergence():
    """Expose an improving feasible iterate for native verification without calling it converged."""
    design = torch.nn.ParameterDict({"x": torch.nn.Parameter(torch.full((3,), .5, dtype=torch.float64))})
    result = constrained_minimize(design, Quadratic(design), lambda output: output["loss"],
        bounds=(0., 1.), constraints=lambda output: output["constraints"][:1],
        constraint_bounds=([1.5], [1.5]), steps=2, ftol=1e-12,
        allow_iteration_limit=True)
    assert result["iteration_limited"] and result["feasible"]
    assert not result["success"]
    assert result["objective_after"] < result["objective_before"]
    assert float(design["x"].sum().detach()) == pytest.approx(1.5, abs=1e-12)
    with pytest.raises(RuntimeError):
        constrained_minimize(design, Quadratic(design), lambda output: output["loss"],
            bounds=(0., 1.), constraints=lambda output: output["constraints"][:1],
            constraint_bounds=([4.], [4.]), steps=1, ftol=1e-12,
            allow_iteration_limit=True)


def test_resumed_candidate_projects_to_known_nonlinear_constraint():
    """Recover the closest point on a unit circle while preserving the reference."""
    initial = torch.tensor([1., 0.], dtype=torch.float64)
    design = torch.nn.ParameterDict({"x": torch.nn.Parameter(initial.clone())})
    checkpoint = np.array([.9, .2])
    result = constrained_minimize(design, lambda: {"x": design["x"]},
        lambda output: (output["x"] - initial.flip(0)).square().sum(),
        bounds=(-2., 2.), constraints=lambda output: output["x"].square().sum().reshape(1),
        constraint_bounds=([1.], [1.]), steps=1, ftol=1e-10,
        allow_iteration_limit=True, starting_point=checkpoint)
    np.testing.assert_allclose(design["x"].detach(), checkpoint / np.linalg.norm(checkpoint),
                               rtol=0., atol=1e-8)
    assert result["objective_before"] == pytest.approx(2.)
    assert result["objective_after"] < result["objective_before"]
    assert result["iteration_limited"] and result["feasible"]
    assert not result["success"]
    assert result["feasibility_correction"]["success"]
    assert result["trace"][-1]["phase"] == "feasibility"
    retained = design["x"].detach().clone()
    with pytest.raises(RuntimeError, match="Constrained design search failed"):
        constrained_minimize(design, lambda: {"x": design["x"]},
            lambda output: output["x"].square().sum(), bounds=(-2., 2.),
            constraints=lambda output: output["x"].square().sum().reshape(1),
            constraint_bounds=([-1.], [-1.]), steps=1, ftol=1e-10,
            allow_iteration_limit=True, starting_point=checkpoint)
    torch.testing.assert_close(design["x"], retained, rtol=0., atol=0.)


def test_best_feasible_iterate_beats_the_last_or_lower_infeasible_point(monkeypatch):
    """Choose the analytical best feasible point from a nonmonotone trial history.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Supply a controlled optimizer trajectory while retaining real physics
        values, constraint evaluation and exact derivatives.
    """
    design = torch.nn.ParameterDict({"x": torch.nn.Parameter(torch.full((3,), .5, dtype=torch.float64))})
    trajectory = np.array([[.9, .1, .6], [.8, .2, .5], [.6, .4, .5]])

    def trial_history(fun, start, *, callback, **options):
        """Supply an infeasible low loss, a feasible improvement and a worse endpoint.

        Parameters
        ----------
        fun : callable
            Real differentiable objective wrapper.
        start : numpy.ndarray
            Original design, unused by this controlled trajectory.
        callback : callable
            Production feasibility and candidate-retention callback.
        **options : dict
            Remaining optimizer arguments, unused by the fixture.

        Returns
        -------
        scipy.optimize.OptimizeResult
            A successful optimizer endpoint that is not its best feasible point.
        """
        for point in trajectory:
            fun(point)
            callback(point)
        return OptimizeResult(x=trajectory[-1], success=True, status=0,
                              message="Controlled nonmonotone trajectory", nit=3)

    monkeypatch.setattr("optimization.constrained.minimize", trial_history)
    result = constrained_minimize(design, Quadratic(design), lambda output: output["loss"],
        bounds=(0., 1.), constraints=lambda output: output["constraints"][:1],
        constraint_bounds=([1.5], [1.5]), steps=10, ftol=1e-12)
    np.testing.assert_allclose(design["x"].detach(), trajectory[1], atol=1e-14)
    assert result["selected_trace_index"] == 2
    assert not result["trace"][1]["feasible"]
    assert result["trace"][1]["objective"] < result["objective_after"]
    assert result["objective_after"] < result["trace"][-1]["objective"]
    assert result["improved"]


class ScaledDesign(torch.nn.Module):
    """Small material-calibration model with a known native multiplier."""

    def __init__(self, design):
        """Bind the design without making it a calibration-owned parameter.

        Parameters
        ----------
        design : torch.nn.ParameterDict
            Two physical coordinates.
        """
        super().__init__()
        object.__setattr__(self, "design", design)
        self.factor = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))

    def forward(self):
        """Expose the calibrated scalar and its physical inventory constraint.

        Returns
        -------
        dict
            Scaled squared norm and the sum of design coordinates.
        """
        values = self.design["x"]
        return {"metric": self.factor * values.square().sum(), "inventory": values.sum()}


def test_explicit_search_keeps_native_rejection_and_restoration():
    """A rejected constrained proposal restores the design in the shared loop."""
    design = torch.nn.ParameterDict({"x": torch.nn.Parameter(torch.tensor([.2, .8], dtype=torch.float64))})
    seen = []

    def detailed(current, previous):
        """Return an independently scaled native response.

        Parameters
        ----------
        current : torch.nn.ParameterDict
            Candidate physical design.
        previous : Snapshot or None
            Unused prior result.

        Returns
        -------
        Snapshot
            Known native response and its calibration target.
        """
        metric = 1.2 * current["x"].square().sum()
        seen.append(current["x"].clone())
        return Snapshot({"metric": metric}, {"observables": {"metric": {"mean": float(metric), "scale": .01}}})

    result = optimize(design, detailed, lambda current, reference: ScaledDesign(current),
        ["metric"], lambda output: output["metric"], steps=1,
        accept=lambda current, candidate, proposal: False,
        fit_options={"steps": 15, "jacobian_mode": "reverse"},
        search=lambda current, model, objective: constrained_minimize(current, model, objective,
            bounds=(0., 1.), constraints=lambda output: output["inventory"].reshape(1),
            constraint_bounds=([1.], [1.]), steps=20, ftol=1e-12))
    torch.testing.assert_close(seen[1], torch.full((2,), .5, dtype=torch.float64), rtol=0, atol=1e-8)
    torch.testing.assert_close(design["x"], torch.tensor([.2, .8], dtype=torch.float64), rtol=0, atol=0)
    assert result["proposals"][0]["accepted"] is False
    assert result["fits"][0]["locally_identifiable"]
    assert result["fits"][0]["loss"] < 1e-10


def test_search_without_a_feasible_improvement_preserves_the_native_state():
    """A stopped local search cannot mutate the accepted design or run transport."""
    initial = torch.tensor([.2, .8], dtype=torch.float64)
    design = torch.nn.ParameterDict({"x": torch.nn.Parameter(initial.clone())})
    reference = Snapshot({"metric": torch.tensor(.816, dtype=torch.float64)},
                         {"observables": {"metric": {"mean": .816, "scale": .01}}})

    def no_candidate(current, model, objective):
        """Emulate an unsuccessful proposal after changing trial coordinates.

        Parameters
        ----------
        current : torch.nn.ParameterDict
            Design whose accepted coordinates must survive this attempt.
        model, objective : callable
            Unused fitted model and scalar objective for this stopping fixture.

        Returns
        -------
        dict
            Explicit absence of an improving feasible proposal.
        """
        with torch.no_grad():
            current["x"].fill_(.5)
        return {"proposed": False, "stop_reason": "no_improving_feasible_candidate"}

    result = optimize(design, lambda *_: pytest.fail("A stopped search must not run native transport"),
        lambda current, reference: ScaledDesign(current), ["metric"], lambda output: output["metric"],
        steps=None, initial_snapshot=reference, accept=lambda *_: True, search=no_candidate,
        fit_options={"steps": 15, "jacobian_mode": "reverse"})
    torch.testing.assert_close(design["x"], initial, rtol=0., atol=0.)
    assert result["snapshot"] is reference
    assert not result["proposals"]
    assert result["stop_reason"] == "no_improving_feasible_candidate"


@pytest.mark.parametrize("starting", [.2, .5])
@pytest.mark.parametrize("steps", [5, None])
def test_native_goal_stops_the_shared_loop_after_the_current_fit(starting, steps):
    """Stop at an actual native target, including when the initial state satisfies it.

    Parameters
    ----------
    starting : float
        First coordinate; 0.5 starts at the known target already.
    steps : int or None
        A finite proposal budget or continuation controlled by the native goal.
    """
    design = torch.nn.ParameterDict({"x": torch.nn.Parameter(torch.tensor([starting, 1-starting], dtype=torch.float64))})
    seen = []

    def detailed(current, previous):
        """Evaluate an independent scalar whose target has nonzero calibration rank.

        Parameters
        ----------
        current : torch.nn.ParameterDict
            Physical coordinates.
        previous : Snapshot or None
            Unused preceding observation.

        Returns
        -------
        Snapshot
            Known native value and its material-fitting observation.
        """
        metric = 1.2 * current["x"].square().sum()
        seen.append(float(metric))
        return Snapshot({"metric": metric}, {"observables": {"metric": {"mean": float(metric), "scale": .01}}})

    result = optimize(design, detailed, lambda current, reference: ScaledDesign(current),
        ["metric"], lambda output: (output["metric"] - .6).square(), steps=steps,
        accept=lambda current, candidate, proposal: True,
        stop=lambda reference: abs(float(reference.outputs["metric"]) - .6) < 1e-6,
        fit_options={"steps": 15, "jacobian_mode": "reverse"},
        search=lambda current, model, objective: constrained_minimize(current, model, objective,
            bounds=(0., 1.), constraints=lambda output: current["x"].new_empty(0),
            constraint_bounds=([], []), steps=30, ftol=1e-14))
    assert seen[-1] == pytest.approx(.6, abs=1e-6)
    assert len(seen) == (1 if starting == .5 else 2)
    assert len(result["proposals"]) == len(seen) - 1
    assert result["fits"][-1]["locally_identifiable"]
