"""Bounded scalar design searches with exact Torch constraint derivatives."""

import time

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize
import torch

from .solvers import frozen


def constrained_minimize(design, model, objective, *, bounds, constraints,
                         constraint_bounds, steps, ftol, callback=None,
                         allow_iteration_limit=False, starting_point=None,
                         feasibility_steps=8):
    """Minimize a scalar response with box and nonlinear physical constraints.

    Parameters
    ----------
    design : torch.nn.Module
        CPU design parameters. Their named-parameter order defines the flattened
        optimization vector; physical units belong to the caller.
    model : callable
        Zero-argument differentiable model returning named responses. Trainable
        module coefficients outside the design are frozen during this search.
    objective : callable
        Scalar Torch objective evaluated from model outputs.
    bounds : tuple of array_like
        Lower and upper design bounds, broadcastable to the flattened vector.
    constraints : callable
        Torch constraint vector evaluated from the same model outputs. Scale
        physical constraints explicitly so their tolerances are meaningful.
    constraint_bounds : tuple of array_like
        Lower and upper bounds for that vector. Equal bounds impose equalities.
    steps : int
        Maximum SLSQP iterations.
    ftol : float
        SLSQP objective and constraint convergence tolerance in the declared
        scaled quantities.
    callback : callable or None, optional
        Receive each accepted iterate's parameter, objective and constraint record.
    allow_iteration_limit : bool, optional
        Return the best feasible iterate when only the iteration budget prevents
        convergence. Tolerances remain ftol. The report distinguishes feasibility,
        improvement and convergence; the caller independently verifies proposals.
    starting_point : array_like or None, optional
        Saved flattened candidate from which to resume. The current design
        remains the reference for improvement and restoration on failure.
    feasibility_steps : int, optional
        Maximum iterations for projecting a budget-limited candidate onto the
        declared constraints. Projection minimizes parameter displacement and
        uses the same bounds and numerical tolerance as the design search.

    Returns
    -------
    dict
        Selected parameters, initial objective gradient for every parameter,
        objective/constraint history, convergence status and measured timings.

    Raises
    ------
    RuntimeError
        SLSQP fails or no feasible candidate is found. The starting design is
        restored on failure. A feasible non-improvement is reported explicitly.

    Notes
    -----
    Every distinct parameter vector needs one forward solve. A backward pass per
    scalar objective or constraint obtains all parameter derivatives together.
    SLSQP receives these derivatives directly; it does not finite-difference the
    physics. Cached values and derivatives share the same forward solution.
    """
    named = [(name, value) for name, value in design.named_parameters() if value.requires_grad]
    parameters = [value for _, value in named]
    initial = torch.cat([value.detach().reshape(-1) for value in parameters]).numpy().copy()
    start = initial if starting_point is None else np.asarray(starting_point, dtype=float)
    if np.any(start < np.asarray(bounds[0])) or np.any(start > np.asarray(bounds[1])):
        raise ValueError("The starting candidate lies outside the declared design bounds")
    shapes = [value.shape for value in parameters]
    ends = np.cumsum([value.numel() for value in parameters])
    starts = np.r_[0, ends[:-1]]
    owned = {id(value) for value in parameters}
    fixed = [value for value in model.parameters() if id(value) not in owned] if isinstance(model, torch.nn.Module) else []
    cache, trace, best = {}, [], {}
    forward_wall, backward_wall = 0., 0.
    evaluations = 0
    phase = "optimization"
    correction = None

    def assign(vector):
        """Copy flattened physical coordinates into their owning parameters.

        Parameters
        ----------
        vector : numpy.ndarray
            Coordinates in named-parameter order and caller-defined units.
        """
        with torch.no_grad():
            for parameter, start, end, shape in zip(parameters, starts, ends, shapes):
                parameter.copy_(torch.as_tensor(vector[start:end]).reshape(shape))

    def values_at(vector):
        """Evaluate a point once and retain exact objective/constraint Jacobians.

        Parameters
        ----------
        vector : numpy.ndarray
            Candidate physical coordinates.

        Returns
        -------
        dict
            Point, response vector and row-by-parameter Jacobian.
        """
        nonlocal forward_wall, backward_wall, evaluations
        if "point" in cache and np.array_equal(vector, cache["point"]):
            return cache
        assign(vector)
        tick = time.perf_counter()
        output = model()
        values = torch.cat((objective(output).reshape(1), constraints(output).reshape(-1)))
        forward_wall += time.perf_counter() - tick
        tick = time.perf_counter()
        rows = []
        for index, value in enumerate(values):
            gradient = torch.autograd.grad(value, parameters, retain_graph=index < len(values) - 1)
            rows.append(torch.cat([item.reshape(-1) for item in gradient]).detach().numpy())
        backward_wall += time.perf_counter() - tick
        evaluations += 1
        cache.update(point=vector.copy(), values=values.detach().numpy(), jacobian=np.asarray(rows))
        return cache

    def is_feasible(vector):
        """Check every declared physical constraint and parameter bound.

        Parameters
        ----------
        vector : numpy.ndarray
            Candidate coordinates in the caller's physical units.

        Returns
        -------
        bool
            Whether all constraints hold within the unchanged ftol.
        """
        values = values_at(vector)["values"][1:]
        return bool(np.all(values >= lower - ftol)
                    and np.all(values <= upper + ftol)
                    and np.all(vector >= np.asarray(bounds[0]) - ftol)
                    and np.all(vector <= np.asarray(bounds[1]) + ftol))

    def record(vector):
        """Retain one accepted optimizer iterate without duplicating its solve.

        Parameters
        ----------
        vector : numpy.ndarray
            Accepted physical coordinates.
        """
        result = values_at(vector)
        entry = {"phase": phase, "parameters": vector.tolist(), "objective": float(result["values"][0]),
                 "constraints": result["values"][1:].tolist(), "feasible": is_feasible(vector)}
        trace.append(entry)
        if entry["feasible"] and (not best or entry["objective"] < best["objective"]):
            best.update(point=vector.copy(), objective=entry["objective"], trace_index=len(trace) - 1)
        if callback is not None:
            callback(entry)

    started = time.perf_counter()
    try:
        with frozen(fixed):
            first = values_at(initial)
            initial_value = float(first["values"][0])
            initial_gradient = first["jacobian"][0].copy()
            lower, upper = [np.broadcast_to(value, first["values"][1:].shape)
                            for value in constraint_bounds]
            equality = lower == upper
            restrictions = [NonlinearConstraint(
                lambda vector, selected=selected: values_at(vector)["values"][1:][selected],
                lower[selected], upper[selected],
                jac=lambda vector, selected=selected: values_at(vector)["jacobian"][1:][selected])
                for selected in (equality, ~equality) if selected.any()]
            record(start)
            result = minimize(lambda vector: values_at(vector)["values"][0], start,
                method="SLSQP", jac=lambda vector: values_at(vector)["jacobian"][0],
                bounds=Bounds(*bounds), constraints=restrictions, callback=record,
                options={"maxiter": steps, "ftol": ftol})
            candidate = result.x
            if result.status == 9 and allow_iteration_limit and not is_feasible(candidate):
                phase = "feasibility"
                projected = minimize(lambda vector: .5 * np.sum((vector - result.x)**2),
                    candidate, method="SLSQP", jac=lambda vector: vector - result.x,
                    bounds=Bounds(*bounds), constraints=restrictions, callback=record,
                    options={"maxiter": feasibility_steps, "ftol": ftol})
                candidate = projected.x
                correction = {"success": bool(projected.success), "message": projected.message,
                              "iterations": int(projected.nit),
                              "displacement_norm": float(np.linalg.norm(candidate - result.x))}
            if not np.array_equal(candidate, trace[-1]["parameters"]):
                record(candidate)
            limited = bool(result.status == 9 and allow_iteration_limit)
            if not best:
                raise RuntimeError("Constrained design search failed to meet the requested constraint tolerance")
            if not result.success and not limited:
                raise RuntimeError(f"Constrained design search failed: {result.message}")
            candidate = best["point"]
            final = values_at(candidate)
            feasible = is_feasible(candidate)
            if not feasible:
                raise RuntimeError("Selected feasible iterate failed re-evaluation at the original tolerance")
            assign(candidate)
    except BaseException:
        assign(initial)
        raise
    return {"optimizer": "SLSQP", "success": bool(result.success), "message": result.message,
            "feasible": feasible, "iteration_limited": limited,
            "improved": bool(final["values"][0] < initial_value - ftol),
            "selected_trace_index": best["trace_index"],
            "selection": "best_feasible_iterate",
            "feasibility_correction": correction,
            "objective_before": initial_value, "objective_after": float(final["values"][0]),
            "gradient": {name: initial_gradient[start:end].reshape(shape).tolist()
                         for (name, _), start, end, shape in zip(named, starts, ends, shapes)},
            "parameters": {name: value.detach().tolist() for name, value in named},
            "constraints": final["values"][1:].tolist(), "trace": trace,
            "optimizer_iterations": int(result.nit), "function_evaluations": evaluations,
            "forward_wall_s": forward_wall, "backward_wall_s": backward_wall,
            "wall_s": time.perf_counter() - started}
