"""Bounded scalar response search using a caller's Torch forward and derivative.

No material, geometry, or concentration convention is embedded in this module.
A surrogate root is a prediction; the detailed evaluator decides acceptance.
"""
import math
import time

import torch

from diffusion_surrogate.solver import tensor


@torch.enable_grad()
def critical_parameter(response, initial, bounds, *, target=1., tolerance=1e-7,
                       max_evaluations=16):
    """Find a bounded scalar response target with safeguarded Newton steps.

    Parameters
    ----------
    response : callable
        Differentiable mapping from one scalar Torch parameter to a finite scalar Torch
        response.
    initial : float
        Initial parameter value within the declared bounds.
    bounds : tuple of float
        Inclusive lower and upper limits in the parameter's units.
    target : float, optional
        Requested scalar response value.
    tolerance : float, optional
        Absolute response error allowed for convergence.
    max_evaluations : int, optional
        Maximum number of response and derivative evaluations.

    Returns
    -------
    dict
        Converged parameter, response, derivative, target, tolerance, evaluation trace and
        elapsed seconds.

    Raises
    ------
    ValueError
        Bounds, controls or response output are invalid.
    ArithmeticError
        The response derivative is nonfinite.
    RuntimeError
        No bounded root is found within the evaluation budget.

    Notes
    -----
    Bounds are hard limits. A failed Newton step uses an available sign bracket; otherwise
    endpoints are evaluated to establish one. Failure to find a root raises instead of
    returning a clipped concentration as a critical state.
    """
    low, high = map(float, bounds)
    initial, target, tolerance = float(initial), float(target), float(tolerance)
    if (not all(math.isfinite(x) for x in (low, high, initial, target, tolerance)) or
            not low < high or not low <= initial <= high or tolerance <= 0 or max_evaluations < 1):
        raise ValueError('Invalid response-search bounds, initial point or tolerance')
    started = time.perf_counter()
    point, trace = initial, []
    for _ in range(max_evaluations):
        x = tensor(point).requires_grad_()
        y = response(x).reshape(())
        if not bool(torch.isfinite(y)):
            raise ValueError('The response must be finite')
        derivative, = torch.autograd.grad(y, x)
        value, slope = float(y.detach()), float(derivative.detach())
        if not math.isfinite(slope):
            raise ArithmeticError('Nonfinite response derivative')
        error = value-target
        trace.append({'parameter': point, 'response': value, 'derivative': slope})
        if abs(error) <= tolerance:
            return {'parameter': point, 'response': value, 'derivative': slope,
                    'target': target, 'tolerance': tolerance, 'trace': trace,
                    'evaluations': len(trace), 'wall_s': time.perf_counter()-started}
        pairs = [(a, b) for a in trace for b in trace if a['parameter'] < b['parameter'] and
                 (a['response']-target)*(b['response']-target) < 0]
        bracket = min(pairs, key=lambda pair: pair[1]['parameter']-pair[0]['parameter']) if pairs else None
        left, right = (bracket[0]['parameter'], bracket[1]['parameter']) if bracket else (low, high)
        candidate = point-error/slope if slope else math.nan
        if not math.isfinite(candidate) or not left < candidate < right or candidate == point:
            if bracket:
                candidate = (left+right)/2
            else:
                missing = [end for end in (low, high) if not any(t['parameter'] == end for t in trace)]
                if not missing:
                    raise RuntimeError('No bracketed root within the supplied parameter bounds')
                candidate = missing[0]
        point = candidate
    raise RuntimeError(f'Response search did not converge in {max_evaluations} evaluations')
