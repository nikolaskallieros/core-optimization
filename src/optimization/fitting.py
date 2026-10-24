"""Generic Torch fitting: a model, selected outputs, and reference results.

The model owns its parameters, constraints, and forward calculation. This module
has no transport equations, geometry assumptions, or parameter-specific updates.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch

from diffusion_surrogate.responses import Response, prediction_values
from runtime.io import save_json


@dataclass(frozen=True)
class Target:
    """Select one observation family and declare its residual transform and weight."""
    name: str
    prediction: object = None  # output key, tensor selection mapping, or callable
    reference: str | None = None
    transform: str = 'log'
    weight: float = 1.

    def resolve(self, output):
        """Select a model output while preserving its declared component ordering.

        Parameters
        ----------
        output : dict
            Named model predictions as Torch tensors or physical Response objects.

        Returns
        -------
        torch.Tensor or Response
            Selected prediction, with an optional tensor-index selection or sum reduction.

        Raises
        ------
        ValueError
            A selection loses Response metadata or uses invalid axes, repeated indices or
            unsupported reduction.

        Notes
        -----
        A JSON selection uses key, indices (one list or null per tensor axis), and reduction
        ('none' or 'sum'). Index lists retain their dimensions and select their Cartesian
        product. Spatial averaging belongs in rebin().
        """
        selection = self.prediction if self.prediction is not None else self.name
        if callable(selection):
            return selection(output)
        if isinstance(selection, str):
            return output[selection]
        value = output[selection['key']]
        if isinstance(value, Response):
            raise ValueError('Select labeled Responses in the model adapter to preserve their layout')
        indices = selection.get('indices', [None]*value.ndim)
        if len(indices) != value.ndim:
            raise ValueError(f'{self.name}: indices must follow the output tensor axes')
        for dim, selected in enumerate(indices):
            if selected is not None:
                if len(set(selected)) != len(selected):
                    raise ValueError(f'{self.name}: repeated axis indices')
                value = value.index_select(dim, torch.tensor(selected, device=value.device))
        reduction = selection.get('reduction', 'none')
        if reduction not in ('none', 'sum'):
            raise ValueError(f'{self.name}: unsupported tensor reduction')
        return value.sum() if reduction == 'sum' else value

    def values(self, output):
        """Return the numerical tensor selected by this calibration target.

        Parameters
        ----------
        output : dict
            Named model predictions as Torch tensors or physical Response objects.

        Returns
        -------
        torch.Tensor
            Selected tensor, unwrapped from Response when needed.
        """
        selected = self.resolve(output)
        return selected.values if isinstance(selected, Response) else selected


def selected_fit(result, *, metadata=None):
    """Create a validated coefficient-only warm-start record.

    Parameters
    ----------
    result : dict
        Completed fitting report with parameter values, objective loss and identifiability
        diagnostics.
    metadata : dict or None, optional
        Caller-owned identity or provenance fields stored with the selected coefficients.

    Returns
    -------
    dict
        Schema-versioned model identity, parameter values, labels and copied metadata.

    Raises
    ------
    ValueError
        The fit is nonfinite, locally unidentifiable or worse than its initial objective.
    """
    if not result['locally_identifiable'] or not math.isfinite(result['loss']):
        raise ValueError('The selected fit must be finite and locally identifiable')
    if result['loss'] > result['initial_loss']+1e-10*max(1., result['initial_loss']):
        raise ValueError('The fitting objective worsened')
    return {key: deepcopy(result[key]) for key in ('model_type', 'parameter_values', 'parameter_labels')} | {
        'schema_version': 1, 'metadata': deepcopy(metadata or {})}


def load_fit(model, selected, *, parameter_labels=None):
    """Restore fitted parameters while retaining current model buffers.

    Parameters
    ----------
    model : torch.nn.Module
        Differentiable model exposing the trainable coefficients to fit.
    selected : dict or str or pathlib.Path
        Selected-fit record or JSON path containing compatible coefficients.
    parameter_labels : dict or None, optional
        Declared coefficient labels and ordering for validating parameter identity.

    Returns
    -------
    dict
        Initialization kind, source path when applicable and retained metadata.

    Raises
    ------
    ValueError
        The selected model identity, parameter labels, names, shapes or values are
        incompatible.
    """
    record = json.loads(Path(selected).read_text()) if isinstance(selected, (str, Path)) else selected
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    model_type = f'{type(model).__module__}.{type(model).__qualname__}'
    if record.get('schema_version') != 1 or record.get('model_type') != model_type:
        raise ValueError('Selected fit belongs to a different model parameterization')
    if record['parameter_labels'] != (parameter_labels or {}):
        raise ValueError('Selected fit parameter labels/order differ')
    if set(record['parameter_values']) != set(parameters):
        raise ValueError('Selected fit parameter names differ')
    values = {}
    for name, parameter in parameters.items():
        value = torch.as_tensor(record['parameter_values'][name], dtype=parameter.dtype, device=parameter.device)
        if value.shape != parameter.shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Invalid selected parameter shape/value: {name}')
        values[name] = value
    # Validate the complete record before changing any live parameter.
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(values[name])
    return {'kind': 'selected_fit', 'source': str(selected) if isinstance(selected, (str, Path)) else None,
            'metadata': record.get('metadata', {})}


def save_fit(result, path, *, metadata=None):
    """Atomically persist selected coefficients for a subsequent calculation.

    Parameters
    ----------
    result : dict
        Completed fitting report with parameter values, objective loss and identifiability
        diagnostics.
    path : str or pathlib.Path
        Destination JSON path for the selected-fit record.
    metadata : dict or None, optional
        Caller-owned identity or provenance fields stored with the selected coefficients.

    Notes
    -----
    Optimizer history, geometry, material data and observation values are excluded.
    Numerical checks here do not replace the caller's design/fit acceptance rule.
    """
    save_json(path, selected_fit(result, metadata=metadata), durable=True)


def fit(model, targets, reference, *, optimizer=torch.optim.LBFGS,
        optimizer_options=None, steps=100, covariance=None, callback=None, rank_rtol=1e-6,
        initial=None, parameter_labels=None, jacobian_mode='reverse', args=(), kwargs=None):
    """Fit trainable coefficients to explicitly scaled model observations.

    Parameters
    ----------
    model : torch.nn.Module
        Differentiable model exposing the trainable coefficients to fit.
    targets : sequence of Target or str
        Unique observation selections with transforms and weights; names use default Target
        settings.
    reference : dict or str or pathlib.Path
        Reference observations or a JSON path; each observable supplies mean and positive
        absolute scale.
    optimizer : type or torch.optim.Optimizer, optional
        Torch optimizer class or an existing instance owning exactly the trainable model
        coefficients.
    optimizer_options : dict or None, optional
        Options used when constructing an optimizer class; supplied instances retain their
        own settings.
    steps : int, optional
        Positive optimization budget; controls internal L-BFGS iterations or ordinary
        optimizer step calls.
    covariance : array_like or None, optional
        Joint covariance of concatenated untransformed observations; None uses the declared
        individual scales.
    callback : callable or None, optional
        Receive closure evaluation count and scalar loss after each backward evaluation.
    rank_rtol : float, optional
        Relative singular-value tolerance for the local parameter-Jacobian rank check.
    initial : dict or str or pathlib.Path or None, optional
        Selected-fit warm start; requires a fresh optimizer without stored curvature.
    parameter_labels : dict or None, optional
        Declared coefficient labels and ordering for validating parameter identity.
    jacobian_mode : {'forward', 'reverse'}, optional
        Automatic-differentiation mode for the final residual Jacobian.
    args : tuple, optional
        Positional inputs passed to model.forward during fitting and Jacobian evaluation.
    kwargs : dict or None, optional
        Keyword inputs passed to model.forward.

    Returns
    -------
    dict
        Fit parameters, predictions, weighted residuals, objective history, Jacobian/rank
        diagnostics, optimizer work counts and timings.

    Raises
    ------
    ValueError
        Targets, scales, prediction layouts, optimizer ownership or warm-start identities
        are invalid.
    ArithmeticError
        The fitting loss becomes nonfinite.

    Notes
    -----
    Reference is a JSON path or mapping with observables[name] entries containing mean and a
    positive absolute scale. Scales are explicit fitting choices; the fitter does not invent
    missing uncertainties. Optional covariance is for the concatenated, untransformed
    selected observations. All target components contribute once; there is no hidden family
    averaging. A supplied Torch optimizer instance is also accepted (its settings prevail).
    initial optionally restores a selected parameter checkpoint. Use a fresh optimizer when
    the reference/geometry changes; prior curvature is not reused. args and kwargs pass
    normal inputs to model.forward, including during the parameter-Jacobian check. Counts
    distinguish optimizer.step() calls, internal L-BFGS iterations, and closure evaluations;
    wall_s includes diagnostics.
    """
    started = time.perf_counter()
    if steps < 1:
        raise ValueError('Fitting steps must be positive')
    if jacobian_mode not in ('forward', 'reverse'):
        raise ValueError('jacobian_mode must be forward or reverse')
    args, kwargs = tuple(args), dict(kwargs or {})
    record = json.loads(Path(reference).read_text()) if isinstance(reference, (str, Path)) else reference
    targets = [Target(t) if isinstance(t, str) else t for t in targets]
    if len({t.name for t in targets}) != len(targets):
        raise ValueError('Choose uniquely named targets')
    parameters = [p for p in model.parameters() if p.requires_grad]
    count = sum(p.numel() for p in parameters)
    if initial is not None and isinstance(optimizer, torch.optim.Optimizer) and optimizer.state:
        raise ValueError('A selected-parameter warm start requires a fresh optimizer')
    initialization = load_fit(model, initial, parameter_labels=parameter_labels) if initial is not None else {'kind': 'model_parameters'}
    initial_parameters = {name: p.detach().tolist() for name, p in model.named_parameters() if p.requires_grad}
    dtype, device = parameters[0].dtype, parameters[0].device
    convert = lambda value: torch.as_tensor(value, dtype=dtype, device=device)
    means, scales, goals, weights, transforms, entries = [], [], [], [], [], []
    for target in targets:
        entry = record['observables'][target.reference or target.name]
        entries.append(entry)
        mean = convert(entry['mean'])
        scale = convert(entry['scale'])
        if scale.numel() != 1 and scale.shape != mean.shape:
            raise ValueError(f'{target.name}: scale must be scalar or match the reference shape')
        mean = mean.reshape(-1)
        scale = scale.reshape(-1).expand_as(mean) if scale.numel() == 1 else scale.reshape(-1)
        if not bool(torch.isfinite(mean).all() & torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError(f'{target.name}: finite means and positive scales are required')
        if target.transform not in ('log', 'identity') or not math.isfinite(target.weight) or target.weight <= 0:
            raise ValueError(f'{target.name}: unsupported transform or weight')
        if target.transform == 'log' and bool((mean <= 0).any()):
            raise ValueError(f'{target.name}: log targets must be positive')
        derivative = 1/mean if target.transform == 'log' else torch.ones_like(mean)
        means.append(mean)
        scales.append(scale*derivative)
        goals.append(mean.log() if target.transform == 'log' else mean)
        weights.append(torch.full_like(mean, target.weight**.5))
        transforms.append(derivative)
    goal, scale, weight = torch.cat(goals), torch.cat(scales), torch.cat(weights)
    if count > goal.numel():
        raise ValueError('More free parameters than selected observations')
    cholesky = None
    if covariance is not None:
        change = torch.cat(transforms)
        cov = convert(covariance)
        if cov.shape != (len(goal), len(goal)):
            raise ValueError('Covariance must match the selected observation order')
        cholesky = torch.linalg.cholesky(change[:, None]*cov*change[None, :])

    def evaluate(output=None):
        """Build weighted transformed residuals for current model predictions.

        Parameters
        ----------
        output : dict or None, optional
            Already evaluated predictions; None evaluates the model with the declared arguments.

        Returns
        -------
        tuple
            Flattened weighted residual vector and the corresponding named model outputs.
        """
        output = model(*args, **kwargs) if output is None else output
        values = []
        for target, entry in zip(targets, entries):
            value = prediction_values(target.resolve(output), entry).reshape(-1)
            if not bool(torch.isfinite(value.detach()).all()):
                raise ValueError(f'{target.name}: nonfinite prediction')
            if target.transform == 'log' and bool((value.detach() <= 0).any()):
                raise ValueError(f'{target.name}: predicted log target is nonpositive')
            values.append(value.log() if target.transform == 'log' else value)
        difference = torch.cat(values)-goal
        residual = (difference/scale if cholesky is None else
                    torch.linalg.solve_triangular(cholesky, difference[:, None], upper=False)[:, 0])
        return weight*residual, output

    if isinstance(optimizer, torch.optim.Optimizer):
        opt = optimizer
        supplied = {id(p) for group in opt.param_groups for p in group['params']}
        if supplied != {id(p) for p in parameters}:
            raise ValueError('Optimizer parameters must be the model trainable parameters')
    else:
        options = dict(optimizer_options or {})
        if issubclass(optimizer, torch.optim.LBFGS):
            defaults = dict(lr=1., max_iter=steps, max_eval=steps*2, history_size=20,
                            line_search_fn='strong_wolfe', tolerance_grad=1e-8, tolerance_change=1e-12)
        else:
            defaults = dict(lr=.03)
        opt = optimizer(parameters, **(defaults | options))
    trace = []

    def closure():
        """Evaluate the fitting loss, populate gradients and record progress.

        Returns
        -------
        torch.Tensor
            Scalar sum of squared weighted residuals.
        """
        opt.zero_grad(set_to_none=True)
        residual, _ = evaluate()
        loss = residual.square().sum()
        if not bool(torch.isfinite(loss.detach())):
            raise ArithmeticError('Nonfinite fitting loss')
        loss.backward()
        trace.append(float(loss.detach()))
        if callback is not None:
            callback(len(trace), trace[-1])
        return loss

    optimizer_steps = 1 if isinstance(opt, torch.optim.LBFGS) else steps
    first_parameter = next(p for group in opt.param_groups for p in group['params'])
    iterations_before = int(opt.state.get(first_parameter, {}).get('n_iter', 0))
    optimization_started = time.perf_counter()
    for _ in range(optimizer_steps):
        opt.step(closure)
    optimization_wall = time.perf_counter()-optimization_started
    optimizer_iterations = (int(opt.state.get(first_parameter, {}).get('n_iter', 0))-iterations_before
                            if isinstance(opt, torch.optim.LBFGS) else optimizer_steps)
    residual, output = evaluate()
    if jacobian_mode == 'forward':
        # Spatial fits have many observations and few coefficients. Calculate
        # all parameter directions together instead of a reverse per bin.
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        point = torch.cat([p.detach().reshape(-1) for _, p in named])

        def residual_at(value):
            """Evaluate residuals with a functional replacement coefficient vector.

            Parameters
            ----------
            value : torch.Tensor
                Flattened trainable coefficients in named-parameter order.

            Returns
            -------
            torch.Tensor
                Weighted residual vector used for forward-mode Jacobian calculation.
            """
            selected, offset = {}, 0
            for name, parameter in named:
                selected[name] = value[offset:offset+parameter.numel()].reshape(parameter.shape)
                offset += parameter.numel()
            return evaluate(torch.func.functional_call(model, selected, args, kwargs))[0]

        jacobian = torch.autograd.functional.jacobian(residual_at, point, vectorize=True,
                                                      strategy='forward-mode').detach()
    else:
        rows = []
        for value in residual:
            derivative = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                                   for p, g in zip(parameters, derivative)]))
        jacobian = torch.stack(rows).detach()
    singular = torch.linalg.svdvals(jacobian)
    rank = int(torch.linalg.matrix_rank(jacobian, rtol=rank_rtol))
    predictions = {}
    for target, mean in zip(targets, means):
        value = target.values(output)
        predictions[target.name] = {'reference': mean.tolist(), 'prediction': value.detach().reshape(-1).tolist()}
    return {'optimizer': type(opt).__name__, 'free_parameters': count, 'selected_components': len(goal),
            'loss': float(residual.detach().square().sum()), 'initial_loss': trace[0],
            'residual': residual.detach().tolist(), 'predictions': predictions,
            'parameter_values': {name: p.detach().tolist() for name, p in model.named_parameters() if p.requires_grad},
            'model_type': f'{type(model).__module__}.{type(model).__qualname__}',
            'parameter_labels': parameter_labels or {}, 'initialization': initialization,
            'initial_parameter_values': initial_parameters,
            'jacobian': jacobian.tolist(), 'jacobian_singular_values': singular.tolist(),
            'jacobian_mode': jacobian_mode,
            'jacobian_rank': rank, 'locally_identifiable': rank == count,
            'rank_relative_tolerance': rank_rtol,
            'jacobian_condition_number': float(singular[0]/singular[-1]) if rank == count else None,
            'gradient_inf': float((2*jacobian.T@residual.detach()).abs().max()),
            'function_evaluations': len(trace), 'loss_trace': trace,
            'optimizer_steps': optimizer_steps, 'optimizer_iterations': optimizer_iterations,
            'optimization_wall_s': optimization_wall, 'wall_s': time.perf_counter()-started,
            'weighting': 'Explicit target scales and weights; joint covariance used.' if covariance is not None else
                         'Explicit target scales and weights; no joint statistical independence is asserted.'}


def native_targets(reference, choices):
    """Build fitting targets from native observables and explicit noise floors.

    Parameters
    ----------
    reference : dict
        Native observations with mean, uncertainty scale and combined_keff fields.
    choices : sequence of tuple
        Entries containing output name, native source name, selected region indices,
        relative floor and family weight.

    Returns
    -------
    targets : list of Target
        Identity-transform observation selectors with per-component weights.
    reference : dict
        Selected observation means, explicit scales and their documented floor policy.

    Notes
    -----
    choices contains (name, native source, selected indices, relative floor, weight). Floors
    are fitting choices, not confidence intervals.
    """
    import numpy as np
    observations, targets = {}, []
    for name, source, selected, relative_floor, weight in choices:
        entry = reference["observables"][source]
        mean, sd = np.asarray(entry["mean"]), np.asarray(entry["scale"])
        if name == "keff":
            mean, sd = map(np.asarray, reference["combined_keff"])
        if selected is not None:
            mean, sd = mean[selected], sd[selected]
        absolute_floor = relative_floor * max(float(abs(mean).max()), 1e-30) * 1e-4
        scale = np.maximum(sd, np.maximum(relative_floor * abs(mean), absolute_floor))
        selection = name
        if selected is not None and name != "fuel_power_W":
            selection = {"key": name, "indices": [selected] + [None] * (mean.ndim - 1)}
        observations[name] = {"mean": mean.tolist(), "scale": scale.tolist()}
        targets.append(Target(name, prediction=selection, transform="identity", weight=weight / mean.size))
    return targets, {"observables": observations, "scale_basis": {
        "relative_floors": {choice[0]: choice[3] for choice in choices},
        "zero_bin_absolute_floor": "1e-4 times family maximum times relative floor",
        "statistical_interpretation": "No independent-observation or confidence-interval claim; scales are fitting choices."}}
