"""Overshoot, recovery and restart checks without finite differences or transport."""
from copy import deepcopy

import pytest
import torch

import coupling.optimization as optimization
from coupling.optimization import backtrack_step
from optimization.design import parameters, set_parameters


class Design(torch.nn.Module):
    def __init__(self):
        """Create four scalar parameters for backtracking checks."""
        super().__init__()
        for name in ("pitch_mm", "element_radius_mm", "enrichment", "cycle_days"):
            self.register_parameter(name, torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)))


class Objective(torch.nn.Module):
    def __init__(self, design):
        """Bind a quadratic target and count fixture forward evaluations.

        Parameters
        ----------
        design : torch.nn.Module
            Live design parameters used by the fixture.
        """
        super().__init__()
        self.design, self.target, self.calls = design, 1., 0

    def forward(self):
        """Evaluate a manufactured quadratic and criticality diagnostic.

        Returns
        -------
        torch.Tensor
            Scalar objective with connected zero sensitivities for unused parameters.
        """
        self.calls += 1
        x = self.design.pitch_mm[0]
        loss = (x - self.target).square()
        loss = loss + sum(p.sum() * 0 for p in self.design.parameters())
        self.latest = {"objective": float(loss.detach()),
                       "metrics": {"loss": float(loss.detach())},
                       "minimum_keff": .99 + .01 * float(x.detach()),
                       "minimum_corrected_keff": .99 + .01 * float(x.detach())}
        return loss


def oversized_trial(learning_rate=4.):
    """Create a completed Adam proposal that overshoots its target.

    Parameters
    ----------
    learning_rate : float, optional
        Adam step scale used to create the deliberately oversized move.

    Returns
    -------
    tuple
        Design, objective, optimizer and restartable pending-step record.
    """
    design = Design()
    model = Objective(design)
    optimizer = torch.optim.Adam([{"params": [p], "lr": learning_rate} for p in design.parameters()])
    loss = model()
    pending = {"design_before": parameters(design), "prediction_before": deepcopy(model.latest),
               "optimizer_before": deepcopy(optimizer.state_dict())}
    loss.backward()
    gradient = {name: p.grad.tolist() for name, p in design.named_parameters()}
    optimizer.step()
    model()
    pending.update(row={"step": 1, "design": parameters(design),
                        "prediction": deepcopy(model.latest), "gradient": gradient},
                   optimizer_after=deepcopy(optimizer.state_dict()))
    return design, model, optimizer, pending


def run_control(design, model, optimizer, pending, maximum_backtracks=4):
    """Backtrack the manufactured proposal while recording checkpoints.

    Parameters
    ----------
    design : torch.nn.Module
        Live design parameters used by the fixture.
    model : Objective
        Manufactured objective with latest prediction diagnostics.
    optimizer : torch.optim.Adam
        Optimizer carrying the proposed step's moments.
    pending : dict
        Before/after design, prediction and optimizer records.
    maximum_backtracks : int, optional
        Maximum additional forward-only trial evaluations.

    Returns
    -------
    tuple
        Final step-control row and copies of saved pending checkpoints.
    """
    config = {"surrogate_keff_target": 1.002,
              "step_control": {"maximum_backtracks": maximum_backtracks, "minimum_fraction": .01}}
    saved = []
    row = backtrack_step(design, model, optimizer, config, pending, lambda d: None,
                         lambda **kwargs: None, lambda: saved.append(deepcopy(pending)))
    return row, saved


def test_overshoot_reuses_gradient_and_retains_rejected_trial():
    """Verify overshoot reuses gradient and retains rejected trial."""
    design, model, optimizer, pending = oversized_trial()
    original = deepcopy(pending["row"])
    calls = model.calls
    row, saved = run_control(design, model, optimizer, pending)
    assert row["prediction"]["objective"] < pending["prediction_before"]["objective"]
    assert row["prediction"]["minimum_corrected_keff"] == pytest.approx(1.002)
    assert row["gradient"] == original["gradient"]
    assert row["step_control"]["trials"][0]["prediction"] == original["prediction"]
    assert row["step_control"]["fraction"] == pytest.approx(.3)
    assert row["step_control"]["optimizer_moments_reset"]
    assert not optimizer.state
    assert model.calls == calls + 1 and len(saved) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.2)


def test_completed_backtrack_resumes_without_repeating_forward_evaluation():
    """Verify completed backtrack resumes without repeating forward evaluation."""
    design, model, optimizer, pending = oversized_trial()
    first, _ = run_control(design, model, optimizer, pending)
    calls = model.calls
    second, saved = run_control(design, model, optimizer, pending)
    assert model.calls == calls and not saved
    assert second == first
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.2)


def test_no_improving_trial_restores_the_previous_design_and_optimizer():
    """Verify no improving trial restores the previous design and optimizer."""
    design, model, optimizer, pending = oversized_trial()
    model.target = 0.
    set_parameters(design, pending["design_before"])
    model()
    pending["prediction_before"] = deepcopy(model.latest)
    set_parameters(design, pending["row"]["design"])
    model()
    pending["row"]["prediction"] = deepcopy(model.latest)
    calls = model.calls
    row, _ = run_control(design, model, optimizer, pending, maximum_backtracks=2)
    assert not row["step_control"]["accepted"]
    assert parameters(design) == pending["design_before"]
    assert model.latest == pending["prediction_before"]
    assert optimizer.state_dict() == pending["optimizer_before"]
    assert model.calls == calls + 2


def test_improving_full_step_needs_no_extra_evaluation_or_moment_reset():
    """Verify improving full step needs no extra evaluation or moment reset."""
    design, model, optimizer, pending = oversized_trial(learning_rate=.5)
    calls = model.calls
    row, saved = run_control(design, model, optimizer, pending)
    assert row["step_control"]["accepted"] and row["step_control"]["fraction"] == 1
    assert not row["step_control"]["optimizer_moments_reset"]
    assert optimizer.state and optimizer.param_groups[0]["lr"] == .5
    assert model.calls == calls and not saved


def test_pending_completed_step_is_replaced_without_repeating_its_backward_pass(monkeypatch, tmp_path):
    """Verify pending completed step is replaced without repeating its backward pass.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped replacements for dependencies or environment settings.
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    design, model, optimizer, pending = oversized_trial()
    original = deepcopy(pending["row"])
    batch = {"anchor_design": pending["design_before"], "steps": [deepcopy(original)]}
    checkpoint = {"phase": "surrogate", "completed_steps": 1, "selected_step": 0,
                  "current_design": original["design"], "current_optimizer": pending["optimizer_after"],
                  "best_design": pending["design_before"], "best_optimizer": pending["optimizer_before"],
                  "best_prediction": pending["prediction_before"], "pending_step": pending}
    config = {"steps_per_refresh": 1, "surrogate_keff_target": 1.002,
              "learning_rates": {name: 4. for name, _ in design.named_parameters()},
              "step_control": {"maximum_backtracks": 4, "minimum_fraction": .01}}
    monkeypatch.setattr(optimization, "CycleObjective", lambda *args: model)
    monkeypatch.setattr(optimization, "project", lambda *args: None)
    calls = model.calls
    optimization.optimize_batch(tmp_path, None, design, config, batch, checkpoint,
                                tmp_path / "optimizer.pt", lambda **kwargs: None, lambda: None)
    assert model.calls == calls + 1
    assert checkpoint["phase"] == "native" and checkpoint["selected_step"] == 1
    assert "pending_step" not in checkpoint and len(batch["steps"]) == 1
    assert batch["steps"][0]["gradient"] == original["gradient"]
    assert batch["steps"][0]["step_control"]["trials"][0]["prediction"] == original["prediction"]
    assert batch["proposal"] == parameters(design)
    restored = torch.load(tmp_path / "optimizer.pt", weights_only=True)
    assert restored["best_design"] == batch["proposal"]
