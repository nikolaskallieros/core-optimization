"""Differentiable metrics with explicit physical weights and no chosen tradeoff.

Burnup and its target/bound must share units (for example MWd/kg initial heavy
metal); initial-heavy-metal masses share one mass unit. Power uses W or any
consistent unit. Time weights are nonnegative quadrature weights in one time
unit. Orbit multiplicities count complete physical channels/bundles: a central
channel has multiplicity one, not six. These metrics do not infer symmetry,
criticality, geometric feasibility, equilibrium, or an acceptable burnup bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from diffusion_surrogate.solver import tensor


def _tensor(value, name, *, like=None):
    """Convert finite objective inputs without discarding tensor graphs.

    Parameters
    ----------
    value : array_like or torch.Tensor
        Numerical objective data to validate.
    name : str
        Field name included in diagnostics.
    like : torch.Tensor or None, optional
        Reference tensor defining the floating dtype and device; None selects float64.

    Returns
    -------
    torch.Tensor
        Validated floating tensor with the requested dtype and device.
    """
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be explicitly supplied as real numeric data")
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if like is not None and (value.dtype != like.dtype or value.device != like.device):
            raise ValueError(f"{name} must share dtype and device with its data tensor")
        result = value
    else:
        result = torch.as_tensor(
            value, dtype=torch.float64 if like is None else like.dtype,
            device=None if like is None else like.device,
        )
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _scalar(value, name, *, like):
    """Require a finite scalar matching an objective tensor.

    Parameters
    ----------
    value : array_like or torch.Tensor
        Numerical objective data to validate.
    name : str
        Field name included in diagnostics.
    like : torch.Tensor or None
        Reference tensor defining the floating dtype and device; None selects float64.

    Returns
    -------
    torch.Tensor
        Zero-dimensional floating tensor.
    """
    result = _tensor(value, name, like=like)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return result


def _multiplicities(value, count, *, like, name):
    """Convert physical counts or supply unit multiplicities.

    Parameters
    ----------
    value : array_like or None
        Positive integer physical counts; None selects ones.
    count : int
        Number of represented objects requiring multiplicities.
    like : torch.Tensor or None
        Reference tensor defining the floating dtype and device; None selects float64.
    name : str
        Field name included in diagnostics.

    Returns
    -------
    torch.Tensor
        Positive integer-valued counts using the reference tensor's floating dtype.
    """
    if value is None:
        return torch.ones(count, dtype=like.dtype, device=like.device)
    result = _tensor(value, name, like=like)
    if result.shape != (count,):
        raise ValueError(f"{name} must have one entry per represented bundle/channel")
    if (result <= 0).any() or (result != torch.round(result)).any():
        raise ValueError(f"{name} must contain positive integer physical multiplicities")
    return result


def discharge_burnup_objective(
    burnup, initial_hm_mass, target_burnup, *, multiplicities=None,
):
    """Measure mass-weighted squared deviation from target burnup.

    Parameters
    ----------
    burnup : array_like or torch.Tensor
        Nonnegative burnup values using the same unit as the target or limit.
    initial_hm_mass : array_like or torch.Tensor
        Positive initial heavy-metal masses in any consistent mass unit.
    target_burnup : float or torch.Tensor
        Positive desired discharge burnup in the same unit as the supplied burnup.
    multiplicities : array_like or None, optional
        Physical-bundle counts per discharge record; None means one each.

    Returns
    -------
    torch.Tensor
        Dimensionless scalar mean of (burnup / target_burnup - 1)², weighted by initial mass
        and physical multiplicity.

    Notes
    -----
    One entry represents a discharged physical bundle or identical orbit. Initial
    heavy-metal mass M_i is positive and never replaced by remaining heavy-metal mass. This
    target deviation includes both spread and mean bias. No discharge/time weighting is
    inferred from repeated records.
    """
    burnup = _tensor(burnup, "burnup")
    if burnup.ndim != 1 or not burnup.numel() or (burnup < 0).any():
        raise ValueError("burnup must be a nonempty nonnegative bundle vector")
    mass = _tensor(initial_hm_mass, "initial_hm_mass", like=burnup)
    if mass.shape != burnup.shape or (mass <= 0).any():
        raise ValueError("initial_hm_mass must be positive with one mass per bundle")
    target = _scalar(target_burnup, "target_burnup", like=burnup)
    if target <= 0:
        raise ValueError("target_burnup must be positive")
    mult = _multiplicities(multiplicities, burnup.numel(), like=burnup, name="multiplicities")
    weight = mult * mass
    return (weight * (burnup / target - 1).square()).sum() / weight.sum()


def aggregate_channel_power(bundle_power):
    """Sum complete axial bundle powers into physical channel powers.

    Parameters
    ----------
    bundle_power : array_like or torch.Tensor
        Complete physical bundle powers in W, shaped (time, channel, axial_bundle).

    Returns
    -------
    torch.Tensor
        Power in W with shape (time, channel).

    Notes
    -----
    Data represent complete physical bundles and channels, not fractional wedge-cell powers.
    The caller must reconstruct those before aggregation.
    """
    power = _tensor(bundle_power, "bundle_power")
    if power.ndim != 3 or any(size == 0 for size in power.shape):
        raise ValueError("bundle_power must have nonempty [time, channel, axial] axes")
    if (power < 0).any():
        raise ValueError("bundle_power cannot be negative")
    return power.sum(dim=-1)


def radial_power_objective(bundle_power, time_weights, *, channel_multiplicities=None):
    """Measure time-averaged variation among physical channel powers.

    Parameters
    ----------
    bundle_power : array_like or torch.Tensor
        Complete physical bundle powers in W, shaped (time, channel, axial_bundle).
    time_weights : array_like or torch.Tensor
        Explicit nonnegative quadrature weights with positive total; repeated states receive
        no implicit weight.
    channel_multiplicities : array_like or None, optional
        Positive integer physical-channel counts per represented channel; None means one
        each.

    Returns
    -------
    torch.Tensor
        Dimensionless variance of channel power divided by its multiplicity-weighted spatial
        mean, averaged with explicit time weights.

    Notes
    -----
    At each time, mean and variance use physical channel multiplicities. The time average
    uses explicit quadrature weights; a repeated pre/post-event sample receives only the
    weight the caller supplies. Every sampled state must have positive mean channel power,
    including zero-weight samples.
    """
    power = aggregate_channel_power(bundle_power)
    weights = _tensor(time_weights, "time_weights", like=power)
    if weights.shape != (power.shape[0],) or (weights < 0).any():
        raise ValueError("time_weights must be a nonnegative vector over times")
    if weights.sum() <= 0:
        raise ValueError("time_weights must have positive total weight")
    mult = _multiplicities(
        channel_multiplicities, power.shape[1], like=power,
        name="channel_multiplicities",
    )
    mean = (power * mult).sum(dim=1) / mult.sum()
    if (mean <= 0).any():
        raise ValueError("Every power sample needs a positive physical channel mean")
    variance = ((power / mean[:, None] - 1).square() * mult).sum(dim=1) / mult.sum()
    return (weights * variance).sum() / weights.sum()


def assembly_power_objective(assembly_power, time_weights, *, assembly_multiplicities=None):
    """Measure time-averaged variation among complete assembly powers.

    Parameters
    ----------
    assembly_power : array_like or torch.Tensor
        Complete physical assembly powers in W with shape (time, assembly).
    time_weights : array_like or torch.Tensor
        Explicit nonnegative quadrature weights with positive total; repeated states receive
        no implicit weight.
    assembly_multiplicities : array_like or None, optional
        Physical assembly counts per represented assembly; None means one each.

    Returns
    -------
    torch.Tensor
        Dimensionless assembly-count-weighted variance, averaged with explicit time weights.

    Notes
    -----
    The input has [time, assembly] axes and complete-assembly powers, not sector-integrated
    powers or power densities. Unequal element radii do not change assembly-count weighting.
    """
    power = _tensor(assembly_power, "assembly_power")
    if power.ndim != 2 or any(size == 0 for size in power.shape) or (power < 0).any():
        raise ValueError("Power must have nonnegative, nonempty [time, assembly] axes")
    weights = _tensor(time_weights, "time_weights", like=power)
    if weights.shape != (power.shape[0],) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Time weights must be nonnegative with positive total weight")
    mult = _multiplicities(assembly_multiplicities, power.shape[1], like=power,
                           name="assembly_multiplicities")
    mean = (power * mult).sum(dim=1) / mult.sum()
    if (mean <= 0).any():
        raise ValueError("Every sample needs positive mean assembly power")
    variance = (mult * (power / mean[:, None] - 1).square()).sum(dim=1) / mult.sum()
    return (weights * variance).sum() / weights.sum()


@dataclass(frozen=True)
class ObjectiveMetrics:
    """Separate dimensionless objectives; neither dominates by default."""

    discharge_burnup_mse: torch.Tensor
    radial_power_variance: torch.Tensor


def evaluate_objectives(
    *,
    discharge_burnup,
    discharge_initial_hm_mass,
    target_burnup,
    bundle_power,
    time_weights,
    discharge_multiplicities=None,
    channel_multiplicities=None,
):
    """Evaluate discharge-burnup and radial-power objectives together.

    Parameters
    ----------
    discharge_burnup : array_like or torch.Tensor
        Burnup of each discharged bundle, using the target's unit.
    discharge_initial_hm_mass : array_like or torch.Tensor
        Positive initial heavy-metal masses in a consistent mass unit.
    target_burnup : float or torch.Tensor
        Positive desired discharge burnup in the same unit as the supplied burnup.
    bundle_power : array_like or torch.Tensor
        Complete physical bundle powers in W, shaped (time, channel, axial_bundle).
    time_weights : array_like or torch.Tensor
        Explicit nonnegative quadrature weights with positive total; repeated states receive
        no implicit weight.
    discharge_multiplicities : array_like or None, optional
        Physical-bundle counts per discharge record; None means one each.
    channel_multiplicities : array_like or None, optional
        Positive integer physical-channel counts per represented channel; None means one
        each.

    Returns
    -------
    ObjectiveMetrics
        Separate dimensionless burnup and power objective tensors.
    """
    return ObjectiveMetrics(
        discharge_burnup_objective(
            discharge_burnup, discharge_initial_hm_mass, target_burnup,
            multiplicities=discharge_multiplicities,
        ),
        radial_power_objective(
            bundle_power, time_weights, channel_multiplicities=channel_multiplicities,
        ),
    )


def scalarize(metrics: ObjectiveMetrics, *, burnup_weight, power_weight):
    """Combine the two objectives using explicit, unnormalized weights.

    Parameters
    ----------
    metrics : ObjectiveMetrics
        Burnup and power objective tensors.
    burnup_weight : float or torch.Tensor
        Nonnegative coefficient of the discharge-burnup objective.
    power_weight : float or torch.Tensor
        Nonnegative coefficient of the radial-power objective; the two weights must have
        positive sum.

    Returns
    -------
    torch.Tensor
        Scalar weighted objective retaining both derivative paths.
    """
    burnup = _scalar(metrics.discharge_burnup_mse, "discharge_burnup_mse", like=metrics.discharge_burnup_mse)
    power = _scalar(metrics.radial_power_variance, "radial_power_variance", like=burnup)
    if burnup < 0 or power < 0:
        raise ValueError("Objective metrics cannot be negative")
    wb = _scalar(burnup_weight, "burnup_weight", like=burnup)
    wp = _scalar(power_weight, "power_weight", like=burnup)
    if wb < 0 or wp < 0 or wb + wp <= 0:
        raise ValueError("Scalarization weights must be nonnegative with positive sum")
    return wb * burnup + wp * power


def peak_burnup_constraint(burnup, *, peak_burnup_limit):
    """Compute a normalized peak-burnup constraint on supplied samples.

    Parameters
    ----------
    burnup : array_like or torch.Tensor
        Nonnegative burnup values using the same unit as the target or limit.
    peak_burnup_limit : float or torch.Tensor
        Positive maximum allowed burnup in the input burnup unit.

    Returns
    -------
    torch.Tensor
        Scalar max(burnup) / peak_burnup_limit - 1; nonpositive values satisfy the
        sampled-state limit.

    Notes
    -----
    A positive bound is mandatory and independent of target burnup. This checks only
    supplied states; it does not establish between-sample feasibility.
    """
    burnup = _tensor(burnup, "burnup")
    if not burnup.numel() or (burnup < 0).any():
        raise ValueError("burnup must contain nonnegative sampled values")
    limit = _scalar(peak_burnup_limit, "peak_burnup_limit", like=burnup)
    if limit <= 0:
        raise ValueError("peak_burnup_limit must be positive")
    return burnup.max() / limit - 1


def temporal_power(times, channel_power, multiplicity):
    """Integrate linear power histories and squared deviations exactly.

    Parameters
    ----------
    times : array_like or torch.Tensor
        Strictly increasing sample times in days, with at least two samples.
    channel_power : array_like or torch.Tensor
        Positive physical channel powers in W, shaped (time, channel).
    multiplicity : array_like or torch.Tensor
        Positive physical-channel counts for each represented channel.

    Returns
    -------
    dict
        Channel mean powers and swings in W, multiplicity-weighted and per-channel
        dimensionless variances, relative ranges and limiting sampled times in days.

    Notes
    -----
    Power is linear between supplied time samples. The squared relative deviation uses the
    exact segment integral (a² + ab + b²) / 3, rather than trapezoidal integration of the
    squared endpoints.
    """
    times, power, counts = tensor(times), tensor(channel_power), tensor(multiplicity)
    if (times.ndim != 1 or power.ndim != 2 or power.shape != (len(times), len(counts))
            or len(times) < 2 or not torch.isfinite(times).all() or not torch.isfinite(power).all()
            or not torch.isfinite(counts).all() or bool((times.diff() <= 0).any())
            or bool((power <= 0).any()) or bool((counts <= 0).any())):
        raise ValueError("Require increasing times and positive, finite full-channel powers/counts")
    weights = times.diff() / (times[-1] - times[0])
    mean = (weights[:, None] * (power[:-1] + power[1:]) / 2).sum(0)
    relative = power / mean - 1
    a, b = relative[:-1], relative[1:]
    per_channel = (weights[:, None] * (a.square() + a * b + b.square()) / 3).sum(0)
    minimum, maximum = power.amin(0), power.amax(0)
    swing = maximum - minimum
    return {"variance": (per_channel * counts).sum() / counts.sum(),
            "variance_by_channel": per_channel, "mean_power_W": mean,
            "relative_range_by_channel": swing / mean,
            "minimum_power_W": minimum, "maximum_power_W": maximum,
            "power_swing_W": swing, "maximum_power_swing_W": swing.amax(),
            "limiting_channel_index": swing.argmax(),
            "minimum_time_days": times[power.argmin(0)],
            "maximum_time_days": times[power.argmax(0)]}


def joint_merit(burnup, temporal_variance, bol_keff, reference, *, power_swing_MW=None):
    """Measure fractional gains against one declared native reference.

    Parameters
    ----------
    burnup : torch.Tensor
        Discharge burnup in MWd/tHM.
    temporal_variance : torch.Tensor
        Dimensionless multiplicity-weighted temporal power variance.
    bol_keff : torch.Tensor
        Beginning-of-life effective multiplication factor.
    reference : mapping
        Positive native burnup, power and reactivity baselines, objective weights and
        selected power metric.
    power_swing_MW : torch.Tensor or None, optional
        Largest channel power swing in MW, required when the reference selects that power
        metric.

    Returns
    -------
    torch.Tensor
        Scalar merit with a fractional burnup reward and fractional power and BOL-reactivity
        costs; lower is better.
    """
    minimax = reference.get("power_metric") == "maximum_channel_power_swing_MW"
    power_name = "maximum_channel_power_swing_MW" if minimax else "temporal_variance"
    power_value = power_swing_MW if minimax else temporal_variance
    if power_value is None:
        raise ValueError("The minimax objective requires the absolute worst-channel power swing")
    scales = [reference[name] for name in ("burnup_MWd_tHM", power_name, "bol_reactivity")]
    if bol_keff is None or any(not math.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("Joint optimization requires positive matched reference metrics")
    weight = reference["weights"]
    if any(not math.isfinite(weight[name]) or weight[name] <= 0
           for name in ("burnup", power_name, "bol_reactivity")):
        raise ValueError("All three joint objective weights must be positive")
    return (-weight["burnup"] * (burnup / scales[0] - 1.)
            + weight[power_name] * (power_value / scales[1] - 1.)
            + weight["bol_reactivity"] * ((1. - 1./bol_keff) / scales[2] - 1.))


def relative_power_density(power, volumes):
    """Normalize local power density by its physical volume-weighted average.

    Parameters
    ----------
    power : torch.Tensor
        Region-integrated powers in a common power unit.
    volumes : torch.Tensor
        Matching physical region volumes in a common volume unit.

    Returns
    -------
    torch.Tensor
        Dimensionless local-to-average power densities with the input shape.

    Raises
    ------
    ValueError
        Power and volume axes differ; broadcasting would change the population.
    """
    if power.shape != volumes.shape:
        raise ValueError("Power and volume axes must match")
    return (power / volumes) / (power.sum() / volumes.sum())


def smooth_maximum(values, sharpness):
    """Approximate a maximum with a differentiable log-sum-exp upper bound.

    Parameters
    ----------
    values : torch.Tensor
        Dimensionless responses over the population being optimized.
    sharpness : float
        Positive dimensionless sharpness; larger values approach the exact maximum.

    Returns
    -------
    torch.Tensor
        Scalar smooth maximum, with derivatives for every input response.
    """
    return torch.logsumexp(sharpness * values.reshape(-1), dim=0) / sharpness


def first_crossing(coordinates, values, *, target=1., extrapolate=False):
    """Interpolate the first downward threshold crossing of a history.

    Parameters
    ----------
    coordinates : array_like or torch.Tensor
        Strictly increasing independent-variable samples in a caller-defined unit.
    values : array_like or torch.Tensor
        Response samples in the same unit as target.
    target : float or torch.Tensor, optional
        Downward-crossing threshold.
    extrapolate : bool, optional
        Allow the final decreasing segment to predict a crossing beyond the sampled
        interval.

    Returns
    -------
    crossing : torch.Tensor
        Crossing coordinate in the input coordinate unit.
    right_index : int
        Right endpoint of the selected segment, or zero when the initial value is already
        below target.
    extrapolated : bool
        Whether the crossing lies beyond the supplied samples.

    Raises
    ------
    ValueError
        Input samples are invalid or no permitted downward crossing exists.

    Notes
    -----
    Coordinates, response values and the threshold belong to the caller. Linear
    extrapolation of the last decreasing segment must be requested explicitly. The
    derivative is local to the selected interpolation segment.
    """
    x, y = tensor(coordinates), tensor(values)
    if (x.ndim != 1 or y.shape != x.shape or len(x) < 2 or
            not bool(torch.isfinite(x).all() & torch.isfinite(y).all()) or
            bool((x.diff() <= 0).any())):
        raise ValueError('Crossing requires finite, strictly ordered history coordinates')
    if float(y[0].detach()) <= target:
        return x[0]+0*y[0], 0, False
    below = torch.nonzero(y.detach() <= target).flatten()
    outside = len(below) == 0
    if outside and not extrapolate:
        raise ValueError('The detailed history has not crossed the target')
    right = len(y)-1 if outside else int(below[0])
    left = right-1
    if float((y[right]-y[left]).detach()) >= 0:
        raise ValueError('The terminal response segment is not decreasing')
    fraction = (target-y[left])/(y[right]-y[left])
    return x[left]+fraction*(x[right]-x[left]), right, outside
