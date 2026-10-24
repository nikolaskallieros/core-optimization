"""Small, geometry-independent helpers around ordinary Torch optimizers."""
from contextlib import contextmanager
from copy import deepcopy
import math
import time

import torch


def project_quadratic_bound(values, lower, upper, weights, limit, *, metric=None):
    """Project onto a weighted quadratic limit and component bounds.

    Parameters
    ----------
    values : sequence of float
        Proposed coordinates in caller-defined units.
    lower : sequence of float
        Lower coordinate bounds, including negative values when applicable.
    upper : sequence of float
        Upper coordinate bounds in the same units.
    weights : sequence of float
        Positive coefficients in ``sum(weight * value**2) <= limit``.
    limit : float
        Finite positive upper bound on the weighted quadratic quantity.
    metric : sequence of float or None, optional
        Positive diagonal weights in ``sum((x - values)**2 / metric)``;
        None uses unit weights.

    Returns
    -------
    list of float
        Bounded coordinates on the feasible side of the quadratic limit,
        allowing only 32 floating-point spacings at a limiting box corner.

    Raises
    ------
    ValueError
        Inputs are invalid or the box contains no feasible point.
    ArithmeticError
        A finite multiplier bracket cannot be found.

    Notes
    -----
    The solution is ``clip(values / (1 + multiplier * weights * metric))``.
    Bisection solves the monotone multiplier equation. This is an algebraic
    projection for optimizer proposals, not an autograd operation.
    """
    count = len(values)
    metric = [1.] * count if metric is None else list(metric)
    if not (len(lower) == len(upper) == len(weights) == len(metric) == count):
        raise ValueError("Quadratic projection dimensions differ")
    if any(not math.isfinite(x) for row in (values, lower, upper, weights, metric) for x in row):
        raise ValueError("Nonfinite quadratic projection input")
    if any(hi < lo or weight <= 0 or scale <= 0
           for lo, hi, weight, scale in zip(lower, upper, weights, metric)):
        raise ValueError("Invalid coordinate box, quadratic weights or projection metric")
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("The quadratic limit must be finite and positive")

    def quadratic(point):
        """Evaluate the declared weighted sum of squared coordinates.

        Parameters
        ----------
        point : sequence of float
            Coordinates in the same order as weights.

        Returns
        -------
        float
            Weighted quadratic quantity in the limit's unit.
        """
        return sum(weight * value * value for weight, value in zip(weights, point))

    closest = [max(lo, min(hi, 0.)) for lo, hi in zip(lower, upper)]
    minimum = quadratic(closest)
    if minimum >= limit:
        if minimum > limit + 32 * math.ulp(limit):
            raise ValueError("Box bounds exclude the quadratic limit")
        return closest

    def candidate(multiplier):
        """Evaluate the bounded coordinates for one multiplier.

        Parameters
        ----------
        multiplier : float
            Nonnegative Lagrange multiplier.

        Returns
        -------
        list of float
            Contracted coordinates clipped to their component bounds.
        """
        return [max(lo, min(hi, value / (1. + multiplier * weight * scale)))
                for value, lo, hi, weight, scale in zip(values, lower, upper, weights, metric)]

    current = candidate(0.)
    if quadratic(current) <= limit:
        return current
    left, right = 0., 1.
    while quadratic(candidate(right)) > limit:
        right *= 2.
        if not math.isfinite(right):
            raise ArithmeticError("Cannot bracket the bounded quadratic projection")
    for _ in range(80):
        midpoint = (left + right) / 2.
        if quadratic(candidate(midpoint)) > limit:
            left = midpoint
        else:
            right = midpoint
    return candidate(right)


@contextmanager
def frozen(parameters):
    """Temporarily disable derivatives for a collection of parameters.

    Parameters
    ----------
    parameters : iterable of torch.Tensor
        Parameters whose requires_grad flags are restored on context exit.

    Yields
    ------
    None
        Control while the selected parameters are frozen.
    """
    parameters = list(parameters)
    enabled = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, value in zip(parameters, enabled):
            parameter.requires_grad_(value)


def _optimized_parameters(design, optimizer):
    """Validate optimizer ownership and identify its trainable design subset.

    Parameters
    ----------
    design : torch.nn.Module
        Live design exposing the candidate physical parameters.
    optimizer : torch.optim.Optimizer
        Optimizer selecting distinct trainable parameters from the design.

    Returns
    -------
    dict of str to torch.nn.Parameter
        Named trainable design parameters selected by the optimizer.

    Raises
    ------
    ValueError
        Parameters are duplicated, foreign to the design or all frozen.
    """
    available = dict(design.named_parameters())
    supplied = [p for group in optimizer.param_groups for p in group['params']]
    ids = {id(p) for p in supplied}
    if len(ids) != len(supplied) or not ids.issubset({id(p) for p in available.values()}):
        raise ValueError('Optimizer parameters must be distinct parameters of the design')
    named = {name: p for name, p in available.items() if id(p) in ids and p.requires_grad}
    if not named:
        raise ValueError('The optimizer must select trainable design parameters')
    return named


def minimize(design, model, objective, *, optimizer, steps=1, project=None,
             keep_best=False, callback=None):
    """Minimize a scalar surrogate objective over selected design parameters.

    Parameters
    ----------
    design : torch.nn.Module
        Live design exposing the candidate physical parameters.
    model : callable
        Zero-argument differentiable model returning the outputs consumed by the objective.
    objective : callable
        Map model outputs to a finite scalar Torch objective.
    optimizer : torch.optim.Optimizer
        Optimizer selecting distinct trainable parameters from the design.
    steps : int, optional
        Number of optimizer.step calls; zero evaluates only the initial objective.
    project : callable or None, optional
        Apply project(design, initial_parameters) in place after each optimizer step.
    keep_best : bool, optional
        Restore the lowest evaluated objective and its optimizer state, including the
        initial point.
    callback : callable or None, optional
        Receive step index, parameter/objective record and model outputs after each
        projected step.

    Returns
    -------
    dict
        Selected parameters, initial/final objectives, initial gradient, step history,
        optimizer work counts and timings.

    Raises
    ------
    ValueError
        Optimization controls, selected parameters or objective shape are invalid.
    ArithmeticError
        An objective, gradient or projected design becomes nonfinite.

    Notes
    -----
    design is any nn.Module (including ParameterDict); model is any callable. No geometry,
    material, response name, or objective is assumed. For a Module model, coefficients
    outside design are frozen temporarily. A callable that closes over other models is
    responsible for freezing their coefficients.

    optimizer is an ordinary Torch optimizer instance; it may select a subset of design
    parameters and may include frozen parameters. steps counts step() calls; L-BFGS may
    perform several iterations/evaluations inside each call. project(design,
    initial_parameters) optionally enforces bounds after each step. Constraints needed
    during line searches belong in the parameterization. callback(step, record, output)
    receives the result after each projected step.

    keep_best selects the lowest evaluated objective, including the starting point, and
    restores its optimizer state too. Errors restore the starting parameters and optimizer.
    Buffers and external callback state are caller-owned. This function runs no
    detailed/reference calculation.
    """
    if steps < 0:
        raise ValueError('steps must be nonnegative')
    named = _optimized_parameters(design, optimizer)
    ids = {id(p) for p in named.values()}
    if not all(bool(torch.isfinite(p).all()) for p in named.values()):
        raise ValueError('Design parameters must be finite')

    started = time.perf_counter()
    before = {name: p.detach().clone() for name, p in named.items()}
    optimizer_before = deepcopy(optimizer.state_dict())
    best_parameters, best_optimizer = before, optimizer_before
    first_parameter = next(p for group in optimizer.param_groups for p in group['params'])
    iterations_before = int(optimizer.state.get(first_parameter, {}).get('n_iter', 0))
    initial_loss, best_loss, final_loss = None, None, None
    selected_step, evaluations, forward_evaluations, gradients = 0, 0, 0, {}
    forward_wall, backward_wall = 0., 0.
    trace = []

    def restore(parameters, state):
        """Restore a saved design subset and optimizer state in place.

        Parameters
        ----------
        parameters : dict
            Saved tensors for optimizer-selected parameter names.
        state : dict
            Optimizer state_dict corresponding to those parameter values.
        """
        with torch.no_grad():
            for name, parameter in named.items():
                parameter.copy_(parameters[name])
        optimizer.load_state_dict(state)

    def evaluate():
        """Evaluate and time the current scalar objective and model outputs.

        Returns
        -------
        tuple
            Validated scalar objective tensor and named model outputs.
        """
        nonlocal forward_wall, forward_evaluations
        tick = time.perf_counter()
        output = model()
        loss = objective(output).reshape(())
        forward_wall += time.perf_counter()-tick
        forward_evaluations += 1
        if not bool(torch.isfinite(loss.detach())):
            raise ArithmeticError('Nonfinite objective')
        return loss, output

    def closure():
        """Populate design gradients and retain the first evaluated gradient.

        Returns
        -------
        torch.Tensor
            Finite scalar objective tensor passed to the optimizer.
        """
        nonlocal initial_loss, best_loss, evaluations, gradients, backward_wall
        optimizer.zero_grad(set_to_none=True)
        loss, _ = evaluate()
        if initial_loss is None:
            initial_loss = best_loss = float(loss.detach())
        tick = time.perf_counter()
        if loss.requires_grad:
            loss.backward()
        backward_wall += time.perf_counter()-tick
        for name, parameter in named.items():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise ArithmeticError(f'Nonfinite design derivative: {name}')
        if evaluations == 0:
            gradients = {name: (torch.zeros_like(p) if p.grad is None else p.grad).detach().tolist()
                         for name, p in named.items()}
        evaluations += 1
        return loss

    other = {id(p): p for p in design.parameters() if id(p) not in ids}
    if isinstance(model, torch.nn.Module):
        other.update({id(p): p for p in model.parameters() if id(p) not in ids})
    try:
        with frozen(other.values()):
            if steps == 0:
                with torch.no_grad():
                    value, _ = evaluate()
                initial_loss = best_loss = final_loss = float(value)
            for step in range(1, steps+1):
                optimizer.step(closure)
                with torch.no_grad():
                    if project is not None:
                        project(design, before)
                    if not all(bool(torch.isfinite(p).all()) for p in named.values()):
                        raise ArithmeticError('Nonfinite design after optimizer/projection')
                    value, output = evaluate()
                    final_loss = float(value)
                if final_loss < best_loss:
                    best_loss, selected_step = final_loss, step
                    if keep_best:
                        best_parameters = {name: p.detach().clone() for name, p in named.items()}
                        best_optimizer = deepcopy(optimizer.state_dict())
                entry = {'step': step, 'objective': final_loss,
                         'parameters': {name: p.detach().tolist() for name, p in named.items()}}
                trace.append(entry)
                if callback is not None:
                    callback(step, entry, output)
            # Record work performed before keep_best potentially rewinds history.
            iterations = (int(optimizer.state.get(first_parameter, {}).get('n_iter', 0))-iterations_before
                          if isinstance(optimizer, torch.optim.LBFGS) else steps)
            if keep_best:
                restore(best_parameters, best_optimizer)
                final_loss = best_loss
            else:
                selected_step = steps
    except BaseException:
        restore(before, optimizer_before)
        raise
    return {'optimizer': type(optimizer).__name__, 'optimizer_steps': steps,
            'optimizer_iterations': iterations, 'function_evaluations': evaluations,
            'forward_evaluations': forward_evaluations,
            'objective_before': initial_loss, 'objective_after': final_loss,
            'gradient': gradients, 'selected_step': selected_step,
            'parameters_before': {name: p.tolist() for name, p in before.items()},
            'parameters_after': {name: p.detach().tolist() for name, p in named.items()},
            'trace': trace, 'forward_wall_s': forward_wall, 'backward_wall_s': backward_wall,
            'wall_s': time.perf_counter()-started}


def bounded_linear_projection(delta, normal, metric, lower, upper, target):
    """Project a bounded move toward one linear equality.

    Parameters
    ----------
    delta : torch.Tensor
        Flattened proposed parameter move.
    normal : torch.Tensor
        Constraint derivative in the same flattened parameter order.
    metric : torch.Tensor
        Nonnegative diagonal inverse-metric weights; zero fixes a component at its
        initially clipped proposal value.
    lower : torch.Tensor
        Lower componentwise move bounds.
    upper : torch.Tensor
        Upper componentwise move bounds.
    target : float or torch.Tensor
        Requested normal-dot-move value.

    Returns
    -------
    torch.Tensor
        Bounded projected move; the equality may remain unmet when no free component can
        correct it.
    """
    value = torch.maximum(torch.minimum(delta, upper), lower).clone()
    for _ in range(value.numel() + 1):
        residual = target - normal @ value
        if abs(float(residual)) <= 1e-12:
            break
        direction = residual * normal
        free = ((direction > 0) & (value < upper)) | ((direction < 0) & (value > lower))
        weight = metric * free
        denominator = (weight * normal.square()).sum()
        if float(denominator) <= 0:
            break
        value = torch.maximum(
            torch.minimum(value + residual * weight * normal / denominator, upper), lower
        )
    return value
