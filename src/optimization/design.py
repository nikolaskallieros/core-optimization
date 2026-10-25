"""Torch design steps between independently evaluated high-fidelity snapshots.

Only the current snapshot is fitted/differentiated. The caller owns its detailed
physics, equilibrium convergence, design constraints and acceptance criteria.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import time

import torch

from optimization.fitting import fit, selected_fit
from optimization.solvers import _optimized_parameters, frozen, minimize


def parameters(design):
    """Serialize named design parameters while preserving their shapes.

    Parameters
    ----------
    design : torch.nn.Module
        Live design exposing all trainable physical parameters.

    Returns
    -------
    dict
        Detached scalar or nested-list values keyed by parameter name.
    """
    return {name: p.detach().tolist() for name, p in design.named_parameters()}


def set_parameters(design, values):
    """Restore all named design parameters in place.

    Parameters
    ----------
    design : torch.nn.Module
        Live design exposing all trainable physical parameters.
    values : dict
        Complete parameter mapping with shapes matching the live design.

    Raises
    ------
    ValueError
        A parameter is missing, extra or has a different shape.
    """
    declared = dict(design.named_parameters())
    if set(values) != set(declared):
        raise ValueError("The optimization must retain every declared design parameter")
    with torch.no_grad():
        for name, p in design.named_parameters():
            value = torch.as_tensor(values[name], dtype=p.dtype, device=p.device)
            if value.shape != p.shape:
                raise ValueError(f"Wrong design shape: {name}")
            p.copy_(value)


@dataclass
class Snapshot:
    """Detailed-model outputs, fitting reference and caller-owned continuation state."""
    outputs: dict
    reference: object
    state: object = None
    metadata: dict = field(default_factory=dict)


def optimize(design, detailed, surrogate, targets, objective, *, optimizer=None,
             steps, accept, project=None, fit_options=None, initial_fit=None,
             initial_snapshot=None, callback=None, fit_accept=None,
             surrogate_steps=1, keep_best=False, search=None, stop=None):
    """Alternate surrogate fitting, design proposals and detailed evaluation.

    Parameters
    ----------
    design : torch.nn.Module
        Live design exposing all trainable physical parameters.
    detailed : callable
        Evaluate detailed(design, previous_snapshot) and return a current Snapshot.
    surrogate : callable
        Build a differentiable surrogate from the live design and current detailed snapshot.
    targets : sequence
        Calibration targets supplied to the shared fitting engine.
    objective : callable
        Map common named model outputs to one finite scalar Torch objective to minimize.
    optimizer : torch.optim.Optimizer or None, optional
        Optimizer selecting the trainable design parameters to update. Supply
        either this optimizer or an explicit search callback.
    steps : int or None
        Maximum detailed proposals; zero fits only the current state. None
        continues until the stop predicate or an explicit search stopping result.
    accept : callable
        Decide accept(current_snapshot, candidate_snapshot, proposal_record), returning bool
        or a mapping with an accepted flag.
    project : callable or None, optional
        In-place design projection applied after each optimizer step.
    fit_options : dict or None, optional
        Additional keyword arguments for the fitting engine.
    initial_fit : dict or None, optional
        Fitted coefficient record used to warm-start the first calibration.
    initial_snapshot : Snapshot or None, optional
        Current detailed state; None triggers an initial detailed evaluation.
    callback : callable or None, optional
        Receive event name, step index and payload for fits, proposals and decisions.
    fit_accept : callable or None, optional
        Predicate that must accept each calibration report before optimization proceeds.
    surrogate_steps : int, optional
        Number of inexpensive optimizer steps before each detailed evaluation.
    keep_best : bool, optional
        Retain the best surrogate iterate before requesting detailed verification.
    search : callable or None, optional
        Alternative search(design, model, objective) returning the common
        minimize result fields. This supports constrained searches without
        duplicating native verification or calibration orchestration. Each
        callback invocation owns a fresh optimizer state. A result containing
        proposed=False stops without a native call and restores the current design.
    stop : callable or None, optional
        Stop after fitting when stop(current_native_snapshot) is true. The
        criterion uses the detailed result, including at the initial reference.

    Returns
    -------
    dict
        Final surrogate, detailed snapshot, selected fit, fit/proposal histories, detailed
        timings and design parameters.

    Raises
    ------
    ValueError
        Step counts, trainable parameters or scalar objectives are invalid.
    TypeError
        Detailed or acceptance callbacks return an incompatible object.
    RuntimeError
        A calibration fails the caller's fit-acceptance predicate.

    Notes
    -----
    detailed(design, previous_snapshot) returns a Snapshot. It may run OpenMC equilibrium
    depletion/search and select one state; it must not return a stale reference for a
    changed design. surrogate(design, snapshot) returns an nn.Module whose geometry depends
    on the live design tensors.

    objective accepts the common named outputs of either calculation. project enforces
    caller-defined bounds/trust limits after optimizer.step(). accept receives
    (current_snapshot, candidate_snapshot, proposal_record); it returns a bool or a dict
    containing 'accepted'. No acceptance/noise policy is hidden.

    The coefficients, but not optimizer curvature, warm-start each new fit. A rejected
    proposal restores design and design-optimizer state. The final current state is fitted
    too, so its returned coefficients are current. steps=0 performs that final fit alone,
    including after a saved design step. surrogate_steps selects how many cheap
    optimizer.step() calls precede each detailed evaluation. All optimizers use Torch's
    closure protocol, including L-BFGS. keep_best optionally retains the best surrogate step
    before verification.
    """
    if (steps is not None and steps < 0) or surrogate_steps < 1:
        raise ValueError('Use nonnegative design steps and positive surrogate_steps')
    parameters = [p for p in design.parameters() if p.requires_grad]
    if (optimizer is None) == (search is None):
        raise ValueError('Supply either a Torch optimizer or an explicit search')
    if optimizer is not None:
        _optimized_parameters(design, optimizer)
    options = dict(fit_options or {})
    options.setdefault('jacobian_mode', 'forward')
    reference = initial_snapshot
    fit_initial = initial_fit
    fits, proposals = [], []
    detailed_wall = []
    stop_reason, stopped_search = None, None

    def evaluate(previous):
        """Evaluate and time the detailed model at the current design.

        Parameters
        ----------
        previous : Snapshot or None
            Previous accepted state available for caller-owned continuation.

        Returns
        -------
        Snapshot
            New detailed result, evaluated without constructing a Torch derivative graph.
        """
        started = time.perf_counter()
        with torch.no_grad():
            result = detailed(design, previous)
        detailed_wall.append(time.perf_counter()-started)
        return result

    def scalar(outputs):
        """Validate and reshape the optimization objective to a scalar tensor.

        Parameters
        ----------
        outputs : dict
            Named outputs from the surrogate or detailed calculation.

        Returns
        -------
        torch.Tensor
            Finite zero-dimensional objective tensor.
        """
        value = objective(outputs).reshape(())
        if not bool(torch.isfinite(value.detach())):
            raise ValueError('The objective must be finite')
        return value

    if reference is None:
        reference = evaluate(None)
    step = 0
    while True:
        model = surrogate(design, reference)
        started = time.perf_counter()
        with frozen(parameters):
            report = fit(model, targets, reference.reference, initial=fit_initial, **options)
        report['wall_s'] = time.perf_counter()-started
        chosen = selected_fit(report)
        if fit_accept is not None and not fit_accept(report):
            raise RuntimeError('The current fit does not satisfy the caller\'s acceptance criteria')
        fits.append(report)
        fit_initial = chosen
        if callback is not None:
            callback('fit', step, {'report': report, 'selected_fit': chosen, 'snapshot': reference})
        if steps is not None and step == steps:
            stop_reason = 'proposal_limit'
            break
        if stop is not None and stop(reference):
            stop_reason = 'caller_stop'
            break
        detailed_before = float(scalar(reference.outputs))
        before = {name: value.detach().clone() for name, value in design.named_parameters()}
        optimizer_before = None if optimizer is None else deepcopy(optimizer.state_dict())

        def restore():
            """Restore the design and optimizer state saved before a proposal."""
            with torch.no_grad():
                for name, parameter in design.named_parameters():
                    parameter.copy_(before[name])
            if optimizer is not None:
                optimizer.load_state_dict(optimizer_before)

        search_result = (minimize(design, model, objective, optimizer=optimizer,
                                 steps=surrogate_steps, project=project, keep_best=keep_best)
                         if search is None else search(design, model, objective))
        if not search_result.get('proposed', True):
            restore()
            stopped_search = search_result
            stop_reason = search_result['stop_reason']
            break
        proposal = {'step': step+1, 'design_before': {k: v.tolist() for k, v in before.items()},
                    'design_candidate': {k: v.detach().tolist() for k, v in design.named_parameters()},
                    'gradient': search_result['gradient'], 'surrogate_objective_before': search_result['objective_before'],
                    'surrogate_objective_candidate': search_result['objective_after'],
                    'detailed_objective_before': detailed_before,
                    'solve_wall_s': search_result['forward_wall_s'], 'backward_wall_s': search_result['backward_wall_s'],
                    'optimization': search_result}
        try:
            if callback is not None:
                callback('proposal', step, proposal)
            candidate = evaluate(reference)
            proposal['detailed_objective_candidate'] = float(scalar(candidate.outputs))
            decision = accept(reference, candidate, proposal)
            if isinstance(decision, dict):
                if not isinstance(decision.get('accepted'), bool):
                    raise TypeError('Acceptance record must contain a boolean accepted field')
                proposal.update(decision)
            elif isinstance(decision, bool):
                proposal['accepted'] = decision
            else:
                raise TypeError('Acceptance callback must return a bool or a dict')
        except BaseException:
            restore()
            raise
        proposals.append(proposal)
        if proposal['accepted']:
            reference = candidate
        else:
            restore()
        if callback is not None:
            callback('decision', step, {'proposal': proposal, 'snapshot': reference,
                                         'candidate': candidate})
        step += 1
    return {'model': model, 'snapshot': reference, 'selected_fit': fit_initial,
            'fits': fits, 'proposals': proposals, 'detailed_wall_s': detailed_wall,
            'stop_reason': stop_reason, 'stopped_search': stopped_search,
            'design_parameters': {k: v.detach().tolist() for k, v in design.named_parameters()}}
