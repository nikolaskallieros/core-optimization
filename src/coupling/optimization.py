"""Batched design updates using a connected full-residence surrogate."""

from copy import deepcopy
import json
import os
import time

import numpy as np
import torch

from optimization.solvers import bounded_linear_projection, frozen, minimize
from runtime.io import sha, save_json, save_checkpoint
from optimization.design import parameters, set_parameters
from runtime.processes import interrupt_on_sigterm
from diffusion_surrogate.solver import tensor

from fuel_cycle.residence import J_PER_MWD, replay_history
from coupling.history import cycle_point_keys


def merit(
    burnup, power_variance, objective, optimization=None, *, bol_keff=None, power_swing_MW=None
):
    """Compute the selected burnup, power or reactivity objective.

    Parameters
    ----------
    burnup : float or torch.Tensor
        Mean discharge burnup in MWd/tHM.
    power_variance : float or torch.Tensor
        Dimensionless spatial or temporal power variance selected by the case.
    objective : mapping
        Cycle objective definitions, reference burnup and optional peak-burnup limit.
    optimization : mapping or None, optional
        Optimization-specific objective settings; None uses the cycle burnup/power
        objective.
    bol_keff : float, torch.Tensor or None, optional
        Matched beginning-of-life keff required by BOL-reactivity or joint objectives.
    power_swing_MW : float, torch.Tensor or None, optional
        Largest channel-power swing in MW when selected by a joint objective.

    Returns
    -------
    float or torch.Tensor
        Dimensionless merit to minimize, preserving derivatives for tensor inputs.
    """
    if optimization is not None and "joint_objective" in optimization:
        from optimization.objectives import joint_merit

        return joint_merit(
            burnup,
            power_variance,
            bol_keff,
            optimization["joint_objective"],
            power_swing_MW=power_swing_MW,
        )
    if optimization is not None and optimization.get("objective_kind") == "bol_reactivity":
        if bol_keff is None or optimization["bol_reactivity"]["reference"] <= 0:
            raise ValueError("BOL reactivity requires its matched state and positive reference")
        return (1.0 - 1.0 / bol_keff) / optimization["bol_reactivity"]["reference"]
    if optimization is not None and optimization.get("objective_kind") == "channel_temporal":
        return power_variance / optimization["temporal_variance_scale"]
    if optimization is not None and optimization.get("objective_kind") == "channel_burnup":
        return (
            -burnup / optimization["burnup_scale_MWd_tHM"]
            + optimization["power_weight"] * power_variance
        )
    if optimization is not None and optimization.get("objective_kind") == "power_flatness":
        return power_variance
    return (
        -burnup / objective["reference_discharge_burnup_MWd_tHM"]
        + objective["assembly_power_weight"] * power_variance
    )


def eol_state_key(keys, config):
    """Select the requested cycle's last sampled endpoint.

    Parameters
    ----------
    keys : sequence of str
        Available native-history stage identifiers in the order of associated arrays.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.

    Returns
    -------
    str
        EOL stage key selected by numeric segment index, not by minimum keff.
    """
    prefix = f"cycle{config['eol_keff']['cycle']}/s"
    endpoints = [key for key in keys if key.startswith(prefix) and key.endswith("_end")]
    if not endpoints:
        raise ValueError("The requested EOL state is missing")
    return max(endpoints, key=lambda key: int(key[len(prefix) : -4]))


def bol_state_key(keys, config):
    """Select the requested cycle's post-loading BOL state.

    Parameters
    ----------
    keys : sequence of str
        Available native-history stage identifiers in the order of associated arrays.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.

    Returns
    -------
    str
        Matching cycle BOL key.
    """
    key = f"cycle{config['bol_reactivity']['cycle']}/bol"
    if key not in keys:
        raise ValueError("The requested BOL state is missing")
    return key


def channel_temporal_check(variance, config):
    """Apply the temporal RMS limit to each physical channel.

    Parameters
    ----------
    variance : array_like
        Dimensionless nonnegative temporal power variance for every represented physical
        channel.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.

    Returns
    -------
    dict
        Per-channel RMS values, missed channel indices and the combined pass flag.
    """
    values = np.asarray(variance, dtype=float)
    limit = config["maximum_channel_temporal_rms"]
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or limit <= 0
    ):
        raise ValueError("Invalid channel variances or temporal RMS limit")
    missed = np.flatnonzero(values > limit**2 + 1e-12).tolist()
    return {
        "limit_rms": limit,
        "rms_by_channel": np.sqrt(values).tolist(),
        "maximum_rms": float(np.sqrt(values.max())),
        "missed_channel_indices": missed,
        "passed": not missed,
    }


def check_residence_gradient_trace(trace, history, config):
    """Check that required depletion paths reach the objective.

    Parameters
    ----------
    trace : sequence of mapping
        Backward stage keys and inventory logarithmic-sensitivity norms.
    history : mapping
        Retained cycle schedules, native observations, design values and endpoint fuel
        states.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.

    Returns
    -------
    dict
        Recorded interval count and required/observed sensitive cycles.

    Raises
    ------
    ArithmeticError
        Stage coverage, finite sensitivities or required earlier-cycle dependence is
        missing.
    """
    expected = {s["end"] for c in history["cycles"] for s in c["segments"]}
    observed = {row["stage"] for row in trace}
    norms = np.array([row["inventory_log_sensitivity_norm"] for row in trace])
    if (
        observed != expected
        or len(trace) != len(expected)
        or not np.isfinite(norms).all()
        or (norms < 0).any()
    ):
        raise ArithmeticError(
            "The full-residence inventory gradient graph is incomplete or invalid"
        )
    cycles = [c["cycle"] for c in history["cycles"]]
    required = (
        [c for c in cycles if c < config["bol_reactivity"]["cycle"]]
        if config.get("objective_kind") == "bol_reactivity"
        else cycles
    )
    sensitive = [
        c
        for c in cycles
        if any(
            row["stage"].startswith(f"cycle{c}/") and row["inventory_log_sensitivity_norm"] > 0
            for row in trace
        )
    ]
    if not set(required).issubset(sensitive):
        raise ArithmeticError(
            "No inventory sensitivity reaches the objective from a required preceding cycle"
        )
    return {
        "recorded_intervals": len(trace),
        "required_sensitive_cycles": required,
        "observed_sensitive_cycles": sensitive,
        "basis": "Every interval remains in the graph. BOL reactivity depends on preceding depletion; "
        "its own cycle can have zero objective sensitivity when the later-state constraints are inactive.",
    }


def eol_check(mean, standard_error, config):
    """Check EOL criticality against its declared measured-SE allowance.

    Parameters
    ----------
    mean : float
        Measured or corrected EOL effective multiplication factor.
    standard_error : float
        Nonnegative standard error of the matched native keff estimate.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.

    Returns
    -------
    dict
        Target interval, measured value, tolerance and pass flag.
    """
    policy = config["eol_keff"]
    if not np.isfinite([mean, standard_error]).all() or standard_error < 0:
        raise ValueError("Invalid EOL k-effective measurement")
    tolerance = policy.get("tolerance", policy["tolerance_standard_errors"] * standard_error)
    return {
        "cycle": policy["cycle"],
        "target": policy["target"],
        "keff": float(mean),
        "standard_error": float(standard_error),
        "tolerance": float(tolerance),
        "lower": policy["target"] - tolerance,
        "upper": policy["target"] + tolerance,
        "passed": bool(abs(mean - policy["target"]) <= tolerance),
    }


def design_loss(
    metrics,
    keff,
    peak,
    objective,
    config,
    keff_error_scale=None,
    *,
    eol_keff=None,
    eol_standard_error=None,
    bol_keff=None,
    channel_temporal_variance=None,
):
    """Combine objective merit with all enabled sampled-state penalties.

    Parameters
    ----------
    metrics : mapping of str to torch.Tensor
        Cycle discharge burnup, power variance and optional maximum channel swing.
    keff : torch.Tensor
        State-ordered keff values, with fixed anchor correction if enabled.
    peak : torch.Tensor
        Largest sampled assembly burnup in MWd/tHM across all cycles.
    objective : mapping
        Cycle objective definitions, reference burnup and optional peak-burnup limit.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.
    keff_error_scale : torch.Tensor or None, optional
        Statewise heuristic keff penalty scales; None uses the configured floor.
    eol_keff : torch.Tensor or None, optional
        Matched EOL multiplication factor when an EOL constraint is enabled.
    eol_standard_error : float or None, optional
        Native standard error at the matched EOL state.
    bol_keff : float, torch.Tensor or None, optional
        Matched beginning-of-life keff required by BOL-reactivity or joint objectives.
    channel_temporal_variance : torch.Tensor or None, optional
        Individual dimensionless channel variances when the per-channel limit is enabled.

    Returns
    -------
    loss : torch.Tensor
        Total differentiable objective including active penalties.
    score : torch.Tensor
        Unpenalized merit for the selected objective.
    """
    score = merit(
        metrics["discharge_burnup_MWd_tHM"],
        metrics["power_variance"],
        objective,
        config,
        bol_keff=bol_keff,
        power_swing_MW=metrics.get("maximum_channel_power_swing_MW"),
    )
    deficit = torch.relu(config["surrogate_keff_target"] - keff)
    if config.get("surrogate_keff_policy") == "anchor_corrected_soft":
        scale = (
            keff_error_scale
            if keff_error_scale is not None
            else torch.full_like(keff, config["surrogate_keff_error_scale_floor"])
        )
        # Small model-scale deficits remain eligible; larger deficits cost rapidly more.
        k_penalty = (
            config["surrogate_keff_soft_weight"] * (deficit / scale).square()
            + (
                torch.relu(deficit - scale) / (config["surrogate_keff_large_deficit_scale"] * scale)
            ).square()
        ).max()
    else:
        k_penalty = (deficit.max() / config["surrogate_keff_penalty_scale"]).square()
    peak_limit = objective.get("peak_burnup_MWd_tHM")
    peak_penalty = (
        (
            torch.relu(peak / peak_limit - config["surrogate_peak_burnup_fraction"])
            / config["surrogate_peak_penalty_scale"]
        ).square()
        if peak_limit is not None
        else 0.0
    )
    power_penalty = (
        (
            torch.relu(metrics["power_variance"] - config["surrogate_power_variance_target"])
            / config["surrogate_power_penalty_scale"]
        ).square()
        if "surrogate_power_variance_target" in config
        else 0.0
    )
    minimum_burnup = config.get("minimum_discharge_burnup_MWd_tHM")
    burnup_penalty = (
        (
            torch.relu(minimum_burnup - metrics["discharge_burnup_MWd_tHM"])
            / config["surrogate_burnup_penalty_scale_MWd_tHM"]
        ).square()
        if minimum_burnup is not None
        else 0.0
    )
    eol_penalty = 0.0
    if config.get("eol_keff"):
        if eol_keff is None or eol_standard_error is None:
            raise ValueError(
                "The EOL objective requires its matched state and native standard error"
            )
        policy = config["eol_keff"]
        scale = max(float(eol_standard_error), config["surrogate_keff_error_scale_floor"])
        eol_penalty = policy["penalty_weight"] * ((eol_keff - policy["target"]) / scale).square()
    temporal_penalty = 0.0
    if "maximum_channel_temporal_rms" in config:
        if channel_temporal_variance is None:
            raise ValueError("The per-channel temporal constraint requires every channel variance")
        limit = config["maximum_channel_temporal_rms"] ** 2
        relative_excess = torch.relu(channel_temporal_variance / limit - 1.0)
        temporal_penalty = (
            (relative_excess / config["channel_temporal_variance_penalty_scale"]).square().sum()
        )
    return (
        score
        + k_penalty
        + eol_penalty
        + peak_penalty
        + power_penalty
        + burnup_penalty
        + temporal_penalty,
        score,
    )


def prediction_rank(prediction, config):
    """Rank predicted feasibility before the penalized objective.

    Parameters
    ----------
    prediction : mapping
        Detached surrogate prediction containing objective metrics and statewise criticality
        values.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.

    Returns
    -------
    tuple
        Feasibility rank and objective value; lexicographically smaller is preferred.
    """
    if config.get("objective_kind") not in (
        "power_flatness",
        "channel_burnup",
        "channel_temporal",
        "bol_reactivity",
    ) and "joint_objective" not in config:
        return (0, prediction["objective"])
    minimum_burnup = config.get("minimum_discharge_burnup_MWd_tHM")
    feasible = (
        (
            minimum_burnup is None
            or prediction["metrics"]["discharge_burnup_MWd_tHM"] >= minimum_burnup
        )
        and ("fresh_fuel_mass" not in config or prediction["fresh_fuel_mass_check"]["passed"])
        and (
            config.get("surrogate_keff_policy") == "anchor_corrected_soft"
            or prediction["minimum_keff"] >= config["surrogate_keff_target"]
        )
        and prediction.get("peak_constraint_passed", False)
        and (
            not config.get("eol_keff")
            or eol_check(
                prediction["eol_corrected_keff"], prediction["eol_standard_error"], config
            )["passed"]
        )
        and (
            "maximum_channel_temporal_rms" not in config
            or channel_temporal_check(prediction["channel_temporal_variance"], config)["passed"]
        )
    )
    return (int(not feasible), prediction["objective"])


def keff_correction(observations, predictions, config):
    """Build fixed state-matched offsets and heuristic error scales.

    Parameters
    ----------
    observations : mapping
        Native state keys mapped to keff mean/standard-error pairs.
    predictions : mapping
        The same state keys mapped to raw surrogate keff predictions.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.

    Returns
    -------
    dict
        Per-state offsets, penalty scales, native errors and aggregate anchor residual RMS.

    Notes
    -----
    The penalty scale combines anchor residual RMS and native standard error with a
    configured floor. It is a model-selection heuristic, not a statistical confidence
    interval.
    """
    if set(observations) != set(predictions):
        raise ValueError("The k-effective correction requires every matched native state")
    keys = sorted(observations)
    native = np.array([observations[key]["keff"] for key in keys])
    raw = np.array([predictions[key]["keff"] for key in keys])
    if (
        native.shape != (len(keys), 2)
        or not np.isfinite(native).all()
        or not np.isfinite(raw).all()
        or (native[:, 1] < 0).any()
    ):
        raise ValueError("Invalid matched k-effective data")
    offset = native[:, 0] - raw
    rms = float(np.sqrt(np.mean(offset**2)))
    scale = np.maximum(np.hypot(rms, native[:, 1]), config["surrogate_keff_error_scale_floor"])
    return {
        "offset_by_state": dict(zip(keys, offset.tolist())),
        "error_scale_by_state": dict(zip(keys, scale.tolist())),
        "native_standard_error_by_state": dict(zip(keys, native[:, 1].tolist())),
        "anchor_residual_rms": rms,
        "scope": "Add native(anchor)-surrogate(anchor) at the same state. Hold offsets fixed within the batch and refresh after refitting. The root-sum-square of anchor residual RMS and native SE sets a heuristic selection penalty scale, not a confidence interval. Raw diffusion/depletion physics and native acceptance remain unchanged.",
    }


def corrected_keff(raw, keys, correction):
    """Apply fixed measured offsets without changing keff derivatives.

    Parameters
    ----------
    raw : torch.Tensor
        Raw state-ordered surrogate effective multiplication factors.
    keys : sequence of str
        Available native-history stage identifiers in the order of associated arrays.
    correction : mapping or None
        Fixed statewise native-minus-surrogate offsets, heuristic error scales and native
        standard errors.

    Returns
    -------
    corrected : torch.Tensor
        Raw keff plus each state's fixed offset.
    scale : torch.Tensor or None
        Heuristic statewise error scales, or None when correction is disabled.
    """
    if correction is None:
        return raw, None
    if set(keys) != set(correction["offset_by_state"]):
        raise ValueError("The k-effective correction does not cover the complete cycle")
    offset = raw.new_tensor([correction["offset_by_state"][key] for key in keys])
    scale = raw.new_tensor([correction["error_scale_by_state"][key] for key in keys])
    return raw + offset, scale


def rescore_prediction(prediction, objective, config, correction):
    """Recompute saved prediction scores under a supplied policy.

    Parameters
    ----------
    prediction : mapping
        Detached surrogate prediction containing objective metrics and statewise criticality
        values.
    objective : mapping
        Cycle objective definitions, reference burnup and optional peak-burnup limit.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.
    correction : mapping or None
        Fixed statewise native-minus-surrogate offsets, heuristic error scales and native
        standard errors.

    Returns
    -------
    dict
        Copied prediction with refreshed merit, penalties and constraint diagnostics; no
        solves are performed.
    """
    result = deepcopy(prediction)
    keys = sorted(result["keff_by_state"])
    raw = tensor([result["keff_by_state"][key] for key in keys])
    corrected, scale = corrected_keff(raw, keys, correction)
    values = {k: tensor(v) for k, v in result["metrics"].items()}
    eol_arguments = {}
    if config.get("eol_keff"):
        key = eol_state_key(keys, config)
        index = keys.index(key)
        se = correction["native_standard_error_by_state"][key]
        eol_arguments = {"eol_keff": corrected[index], "eol_standard_error": se}
        result.update(
            eol_state=key,
            eol_keff=float(raw[index]),
            eol_corrected_keff=float(corrected[index]),
            eol_standard_error=se,
            eol_target_check=eol_check(float(corrected[index]), se, config),
        )
    if (config.get("objective_kind") == "bol_reactivity" or "joint_objective" in config):
        key = bol_state_key(keys, config)
        index = keys.index(key)
        eol_arguments["bol_keff"] = corrected[index]
        result.update(
            bol_state=key,
            bol_keff=float(raw[index]),
            bol_corrected_keff=float(corrected[index]),
            bol_corrected_reactivity=float(1.0 - 1.0 / corrected[index]),
        )
    if "maximum_channel_temporal_rms" in config:
        eol_arguments["channel_temporal_variance"] = tensor(result["channel_temporal_variance"])
        result["channel_temporal_check"] = channel_temporal_check(
            result["channel_temporal_variance"], config
        )
    loss, score = design_loss(
        values,
        corrected,
        tensor(result["maximum_peak_burnup_MWd_tHM"]),
        objective,
        config,
        scale,
        **eol_arguments,
    )
    result.update(
        objective=float(loss),
        merit=float(score),
        minimum_corrected_keff=float(corrected.min()),
        corrected_keff_by_state=dict(zip(keys, corrected.tolist())),
    )
    result["metrics"]["loss"] = float(loss)
    peak_limit = objective.get("peak_burnup_MWd_tHM")
    result["peak_burnup_limit_MWd_tHM"] = peak_limit
    result["peak_constraint_passed"] = (
        peak_limit is None or result["maximum_peak_burnup_MWd_tHM"] <= peak_limit
    )
    return result


def discharge_summary(mapping, state):
    """Account for physical discharged assemblies and initial mass.

    Parameters
    ----------
    mapping : FuelMap
        Region-to-physical-channel/axial mapping and symmetry weights.
    state : FuelState or mapping
        Resident endpoint state with energy_j, initial_hm_kg and fuel IDs.

    Returns
    -------
    dict
        Full-core discharge counts, energy in MWd, initial heavy-metal mass in kg and
        mass-weighted burnup in MWd/tHM.
    """

    def array(value):
        """Expose detached fuel-state values as a NumPy array.

        Parameters
        ----------
        value : array_like or torch.Tensor
            Resident-state values to read without retaining derivatives.

        Returns
        -------
        numpy.ndarray
            Array containing the supplied values.
        """
        return value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    energy = array(state.energy_j if hasattr(state, "energy_j") else state["energy_j"])
    mass = array(state.initial_hm_kg if hasattr(state, "initial_hm_kg") else state["initial_hm_kg"])
    ids = state.ids if hasattr(state, "ids") else state["ids"]
    selected = np.flatnonzero(mapping.levels.numpy() == mapping.count - 1)
    multiplier = mapping.physical.symmetry_multiplier
    full_energy = float(energy[selected].sum() * multiplier)
    full_mass = float(mass[selected].sum() * multiplier)
    rows = []
    for i in selected:
        fraction = float(mapping.fractions[i])
        rows.append(
            {
                "id": ids[i],
                "channel": mapping.channels[i],
                "ring": int(mapping.rings[i]),
                "physical_assemblies": int(mapping.physical.multiplicity[i]),
                "initial_hm_kg_per_assembly": float(mass[i] / fraction),
                "burnup_MWd_tHM": float(energy[i] / J_PER_MWD / (mass[i] / 1000)),
            }
        )
    return {
        "physical_assemblies": sum(r["physical_assemblies"] for r in rows),
        "full_core_discharge_energy_MWd": full_energy / J_PER_MWD,
        "full_core_discharge_initial_hm_kg": full_mass,
        "mean_burnup_MWd_tHM": full_energy / J_PER_MWD / (full_mass / 1000),
        "assemblies": rows,
    }


class CycleObjective(torch.nn.Module):
    """Differentiate the declared objective through a complete surrogate fuel residence."""
    def __init__(self, shared, config):
        """Bind a full-residence surrogate to a declared objective policy.

        Parameters
        ----------
        shared : SharedCycle
            Connected residence model with fixed shared correction coefficients and live design
            parameters.
        config : mapping
            Resolved optimization objective, bounds, native acceptance and step-control
            settings.
        """
        super().__init__()
        self.shared, self.config = shared, config
        self.latest = None
        self.keff_correction = None

    def forward(self, *, gradient_trace=None):
        """Differentiate the complete residence and sampled constraints.

        Parameters
        ----------
        gradient_trace : list or None, optional
            Optional collection receiving stage inventory sensitivity summaries during backward.

        Returns
        -------
        dict
            Live loss, metrics, raw sampled keff, peak burnup and final fuel state; detached
            selection diagnostics are retained in latest.
        """
        shared = self.shared
        observed = {}

        def observe(key, inventory, output):
            """Collect keff at one visited history stage.

            Parameters
            ----------
            key : str
                Stage key used to retain the observed multiplication factor.
            inventory : torch.Tensor
                Region-by-chain CPU float64 atom densities in atoms/cm³.
            output : mapping
                Stage outputs including differentiable keff.
            """
            observed[key] = output["keff"]

        metrics, final, cycles, days = replay_history(
            shared.mapping,
            shared.chain,
            shared.design,
            shared.history,
            shared.points,
            shared.config,
            observe=observe,
            gradient_trace=gradient_trace,
        )
        if set(observed) != set(cycle_point_keys(shared.history)):
            raise ArithmeticError("Optimization omitted a native-history state")
        keys = sorted(observed)
        keff = torch.stack([observed[key] for key in keys])
        corrected, error_scale = corrected_keff(keff, keys, self.keff_correction)
        peak = torch.stack([c["metrics"]["peak_burnup_MWd_tHM"] for c in cycles]).max()
        eol_arguments, eol_fields = {}, {}
        if self.config.get("eol_keff"):
            key = eol_state_key(keys, self.config)
            index = keys.index(key)
            se = shared.history["observations"][key]["keff"][1]
            eol_arguments = {"eol_keff": corrected[index], "eol_standard_error": se}
            eol_fields = {
                "eol_state": key,
                "eol_keff": float(keff[index].detach()),
                "eol_corrected_keff": float(corrected[index].detach()),
                "eol_standard_error": se,
                "eol_target_check": eol_check(float(corrected[index].detach()), se, self.config),
            }
        if (self.config.get("objective_kind") == "bol_reactivity" or "joint_objective" in self.config):
            key = bol_state_key(keys, self.config)
            index = keys.index(key)
            eol_arguments["bol_keff"] = corrected[index]
            eol_fields.update(
                bol_state=key,
                bol_keff=float(keff[index].detach()),
                bol_corrected_keff=float(corrected[index].detach()),
                bol_corrected_reactivity=float((1.0 - 1.0 / corrected[index]).detach()),
            )
        if shared.config["objective"].get("power_basis") == "channel_temporal":
            from fuel_cycle.residence import sample_temporal_power

            temporal = sample_temporal_power(shared.mapping, cycles[-1]["samples"])
            variances = temporal["variance_by_channel"].detach().tolist()
            eol_fields.update(
                channel_temporal_variance=variances,
                channel_power_swing_W=temporal["power_swing_W"].detach().tolist(),
                limiting_power_channel=shared.mapping.channel_ids[
                    int(temporal["limiting_channel_index"])
                ],
                channel_power_minimum_time_days=temporal["minimum_time_days"].detach().tolist(),
                channel_power_maximum_time_days=temporal["maximum_time_days"].detach().tolist(),
            )
            if "maximum_channel_temporal_rms" in self.config:
                eol_arguments["channel_temporal_variance"] = temporal["variance_by_channel"]
                eol_fields["channel_temporal_check"] = channel_temporal_check(
                    variances, self.config
                )
        loss, score = design_loss(
            metrics,
            corrected,
            peak,
            shared.config["objective"],
            self.config,
            error_scale,
            **eol_arguments,
        )
        metrics = {**metrics, "loss": loss}
        discharge = discharge_summary(shared.mapping, final)
        peak_limit = shared.config["objective"].get("peak_burnup_MWd_tHM")
        if not np.isclose(
            discharge["mean_burnup_MWd_tHM"],
            float(metrics["discharge_burnup_MWd_tHM"].detach()),
            rtol=1e-12,
        ):
            raise ArithmeticError("Reported burnup and optimized discharge differ")
        self.latest = {
            **eol_fields,
            "objective": float(loss.detach()),
            "merit": float(score.detach()),
            "metrics": {name: float(value.detach()) for name, value in metrics.items()},
            "minimum_keff": float(keff.min().detach()),
            "minimum_corrected_keff": float(corrected.min().detach()),
            "corrected_keff_by_state": {
                key: float(value.detach()) for key, value in zip(keys, corrected)
            },
            "eol_keff_by_cycle": {
                str(c["cycle"]): float(observed[c["segments"][-1]["end"]].detach())
                for c in shared.history["cycles"]
            },
            "maximum_peak_burnup_MWd_tHM": float(peak.detach()),
            "peak_burnup_limit_MWd_tHM": peak_limit,
            "peak_constraint_passed": peak_limit is None or bool(peak.detach() <= peak_limit),
            "keff_by_state": {key: float(observed[key].detach()) for key in keys},
            "discharge": discharge,
            "cycle_days": float(days.detach()),
            "final_cycle_power_W": [
                shared.mapping.physical_power(s["zone_power_w"]).detach().tolist()
                for s in cycles[-1]["samples"]
            ],
        }
        if "fresh_fuel_mass" in self.config:
            from optimization.fuel_constraints import fuel_mass_check

            self.latest["fresh_fuel_mass_check"] = fuel_mass_check(
                parameters(shared.design), self.config["fresh_fuel_mass"]
            )
        return {"loss": loss, "metrics": metrics, "keff": keff, "peak_burnup": peak, "final_state": final}


def project(design, anchor, config, root):
    """Project a design into trust, mass and geometry constraints.

    Parameters
    ----------
    design : torch.nn.Module
        Live design parameters updated in place by optimization and projection.
    anchor : mapping
        Parameter values at the current native calibration anchor.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.
    root : CycleCase
        Resolved case with model factories, input settings and result paths.

    Raises
    ------
    ValueError
        The case rejects the design for an unhandled reason or no admissible contracted move
        is found.
    """
    with torch.no_grad():
        for name, p in design.named_parameters():
            lower, upper = config["bounds"][name]
            width = config["trust_half_width"][name]
            centre = tensor(anchor[name])
            p.copy_(
                torch.maximum(
                    torch.minimum(p, (centre + width).clamp_max(upper)),
                    (centre - width).clamp_min(lower),
                )
            )
        if "fresh_fuel_mass" in config:
            from optimization.fuel_constraints import project_design_fuel_mass

            project_design_fuel_mass(design, anchor, config)
        trial = parameters(design)
        for i in range(16):
            try:
                root.make_case(design_values=parameters(design))
                return
            except ValueError as error:
                if not any(
                    message in str(error)
                    for message in (
                        "channels overlap or cross",
                        "Channel interfaces overlap",
                        "Channel interfaces cross",
                        "Graphite cylinders overlap",
                        "Graphite sleeves overlap",
                    )
                ):
                    raise
                fraction = 0.5 ** (i + 1)
                set_parameters(
                    design,
                    {
                        name: (
                            tensor(anchor[name])
                            + fraction * (tensor(trial[name]) - tensor(anchor[name]))
                        ).tolist()
                        for name in anchor
                    },
                )
        raise ValueError("No geometrically admissible projected step")


def linearize_eol(model, design):
    """Differentiate EOL keff through the full residence at an anchor.

    Parameters
    ----------
    model : CycleObjective
        Connected residence objective storing its latest detached prediction.
    design : torch.nn.Module
        Live design parameters updated in place by optimization and projection.

    Returns
    -------
    dict
        Anchor design, EOL state/value, parameter gradients, elapsed times and source
        identity.
    """
    named = dict(design.named_parameters())
    ids = {id(p) for p in named.values()}
    with frozen(p for p in model.parameters() if id(p) not in ids):
        started = time.perf_counter()
        output = model()
        forward_s = time.perf_counter() - started
        keys = sorted(model.latest["keff_by_state"])
        key = eol_state_key(keys, model.config)
        started = time.perf_counter()
        gradients = torch.autograd.grad(output["keff"][keys.index(key)], tuple(named.values()))
        backward_s = time.perf_counter() - started
    if not all(torch.isfinite(g).all() for g in gradients):
        raise ArithmeticError("The EOL constraint derivative is nonfinite")
    return {
        "design": parameters(design),
        "state": key,
        "corrected_keff": model.latest["eol_corrected_keff"],
        "gradient": {name: g.detach().tolist() for name, g in zip(named, gradients)},
        "forward_wall_s": forward_s,
        "backward_wall_s": backward_s,
        "scope": "PyTorch derivative through every simulated cycle. The statewise k offset is constant within this fitted batch. Reuse the EOL normal within its trust bounds; evaluate the full nonlinear surrogate after every proposal.",
    }


@torch.no_grad()
def project_eol_step(design, before, prediction, anchor, config, optimizer, linearization, root):
    """Balance a proposed EOL change using Adam's diagonal metric.

    Parameters
    ----------
    design : torch.nn.Module
        Live design parameters updated in place by optimization and projection.
    before : mapping
        Parameter values before the current Adam proposal.
    prediction : mapping
        Detached surrogate prediction containing objective metrics and statewise criticality
        values.
    anchor : mapping
        Parameter values at the current native calibration anchor.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.
    optimizer : torch.optim.Adam
        Live Adam optimizer whose moments and learning rates are part of restart state.
    linearization : mapping
        Saved full-residence EOL normal and its anchor diagnostics.
    root : CycleCase
        Resolved case with model factories, input settings and result paths.

    Returns
    -------
    dict
        Linear target residual diagnostics, Adam metric and the projected proposal.

    Notes
    -----
    The design is updated in place and rechecked for trust, fuel-mass and geometry
    constraints. Nonlinear surrogate evaluation and backtracking still determine acceptance.
    """
    named = dict(design.named_parameters())
    metrics = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            variance = state["max_exp_avg_sq"] if group["amsgrad"] else state["exp_avg_sq"]
            denominator = (variance / (1 - group["betas"][1] ** int(state["step"]))).sqrt() + group[
                "eps"
            ]
            metrics[id(p)] = group["lr"] / denominator
    origin = torch.cat([tensor(before[name]).flatten() for name in named])
    proposed = torch.cat([p.flatten() for p in named.values()])
    normal = torch.cat([tensor(linearization["gradient"][name]).flatten() for name in named])
    metric = torch.cat([metrics[id(p)].flatten() for p in named.values()])
    lower = (
        torch.cat(
            [
                (tensor(anchor[name]) - config["trust_half_width"][name])
                .clamp_min(config["bounds"][name][0])
                .flatten()
                for name in named
            ]
        )
        - origin
    )
    upper = (
        torch.cat(
            [
                (tensor(anchor[name]) + config["trust_half_width"][name])
                .clamp_max(config["bounds"][name][1])
                .flatten()
                for name in named
            ]
        )
        - origin
    )
    target = config["eol_keff"]["target"]
    eol_before = prediction["eol_corrected_keff"]
    delta = bounded_linear_projection(
        proposed - origin, normal, metric, lower, upper, target - eol_before
    )
    start = 0
    for p in named.values():
        p.copy_(
            (origin[start : start + p.numel()] + delta[start : start + p.numel()]).reshape(p.shape)
        )
        start += p.numel()
    project(design, anchor, config, root)
    if "fresh_fuel_mass" in config:
        radius_indices, start = [], 0
        for name, p in named.items():
            if name == config["fresh_fuel_mass"]["radius_parameter"]:
                radius_indices = list(range(start, start + p.numel()))
            start += p.numel()
        proposed = torch.cat([p.flatten() for p in named.values()])
        free_metric = metric.clone()
        free_metric[radius_indices] = 0.
        balanced = origin + bounded_linear_projection(
            proposed - origin,
            normal,
            free_metric,
            lower,
            upper,
            target - eol_before,
        )
        start = 0
        for p in named.values():
            p.copy_(balanced[start : start + p.numel()].reshape(p.shape))
            start += p.numel()
        project(design, anchor, config, root)
    actual_delta = torch.cat([p.flatten() for p in named.values()]) - origin
    predicted = float(eol_before + normal @ actual_delta)
    return {
        "target": target,
        "linearized_eol_keff": predicted,
        "linear_target_reached": abs(predicted - target) <= 1e-10,
        "adam_metric": metric.tolist(),
        "full_proposal": parameters(design),
        "scope": "Project the full Adam proposal with the saved autograd EOL normal, preserving box and geometry bounds. Nonlinear EOL and objective checks still control acceptance and backtracking.",
    }


def native_power_profile(history, mapping, *, include_temporal=False):
    """Reconstruct physical assembly/channel powers and verify totals.

    Parameters
    ----------
    history : mapping
        Retained cycle schedules, native observations, design values and endpoint fuel
        states.
    mapping : FuelMap
        Region-to-physical-channel/axial mapping and symmetry weights.
    include_temporal : bool, optional
        Include the complete final-cycle sampled temporal power profile.

    Returns
    -------
    dict
        Midpoint and time-mean powers in W, physical multiplicities, normalized duration
        weights and recomputed spatial variances.
    """
    cycle = history["cycles"][-1]
    weights = np.array([s["duration_days"] / cycle["cycle_days"] for s in cycle["segments"]])
    power = np.array([history["observations"][s["mid"]]["power_W"] for s in cycle["segments"]])
    power = power / mapping.fractions.numpy()[None, :]
    counts = np.asarray(mapping.physical.multiplicity)
    total = power @ counts
    if (
        not np.isfinite(power).all()
        or (power < 0).any()
        or (total <= 0).any()
        or not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=0.0)
        or not np.allclose(total, total[0], rtol=1e-10, atol=0.0)
    ):
        raise ArithmeticError("Native assembly powers do not close the prescribed cycle power")
    mean = total / counts.sum()
    variance = float(weights @ (((power / mean[:, None] - 1.0) ** 2) @ counts / counts.sum()))
    indices = [i.numpy() for i in mapping.channel_indices]
    channel_power = np.stack([power[:, i].sum(1) for i in indices], axis=1)
    channel_counts = np.array([counts[i[0]] for i in indices])
    channel_mean = channel_power @ channel_counts / channel_counts.sum()
    channel_variance = float(
        weights
        @ (
            ((channel_power / channel_mean[:, None] - 1.0) ** 2)
            @ channel_counts
            / channel_counts.sum()
        )
    )
    expected = cycle["metrics"].get("assembly_power_variance", cycle["metrics"]["power_variance"])
    if not np.isclose(variance, expected, rtol=1e-12, atol=1e-14):
        raise ArithmeticError(
            "Native power profile does not reproduce the optimized flatness metric"
        )
    if "channel_power_variance" in cycle["metrics"] and not np.isclose(
        channel_variance, cycle["metrics"]["channel_power_variance"], rtol=1e-12, atol=1e-14
    ):
        raise ArithmeticError(
            "Native channel powers do not reproduce the optimized flatness metric"
        )
    profile = {
        "fuel_labels": list(mapping.physical.labels),
        "physical_multiplicity": counts.tolist(),
        "duration_fraction": weights.tolist(),
        "midpoint_power_W": power.tolist(),
        "mean_power_W": (weights @ power).tolist(),
        "full_core_power_W": float(weights @ total),
        "power_variance_recomputed": variance,
        "channel_ids": list(mapping.channel_ids),
        "channel_physical_multiplicity": channel_counts.tolist(),
        "channel_midpoint_power_W": channel_power.tolist(),
        "channel_mean_power_W": (weights @ channel_power).tolist(),
        "channel_power_variance_recomputed": channel_variance,
    }
    if include_temporal:
        from fuel_cycle.residence import native_temporal_profile

        profile["temporal"] = native_temporal_profile(history, mapping)
    return profile


def assess_native(result, config):
    """Update native acceptance flags under the current constraints.

    Parameters
    ----------
    result : dict
        Native result updated in place while preserving its original physics observations.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.
    """
    result.setdefault("physics_passed", result["feasible"])
    # A mass-constrained search has no implicit burnup floor.
    minimum_burnup = config.get(
        "minimum_discharge_burnup_MWd_tHM", None if "fresh_fuel_mass" in config else 0.0
    )
    result["burnup_passed"] = (
        minimum_burnup is None or result["metrics"]["discharge_burnup_MWd_tHM"] >= minimum_burnup
    )
    result["bounds_passed"] = all(
        lower <= value <= upper
        for name, (lower, upper) in config["bounds"].items()
        for value in result["design"][name]
    )
    result["flatness_passed"] = (
        result["metrics"]["power_variance"]
        <= config.get("maximum_native_power_variance", float("inf")) + 1e-12
    )
    if "maximum_channel_temporal_rms" in config:
        result["channel_temporal_check"] = channel_temporal_check(
            result["power_profile"]["temporal"]["variance_by_channel"], config
        )
        result["flatness_passed"] = bool(
            result["flatness_passed"] and result["channel_temporal_check"]["passed"]
        )
    eol_passed = True
    if config.get("eol_keff"):
        mean, se = result["eol_keff_by_cycle"][str(config["eol_keff"]["cycle"])]
        result["eol_target_check"] = eol_check(mean, se, config)
        eol_passed = result["eol_target_check"]["passed"]
    mass_passed = True
    if "fresh_fuel_mass" in config:
        from optimization.fuel_constraints import fuel_mass_check

        mass = fuel_mass_check(result["design"], config["fresh_fuel_mass"])
        native_mass = result.get("fresh_fuel_mass_check", mass)
        if not np.isclose(native_mass["fuel_mass_kg"], mass["fuel_mass_kg"], rtol=1e-11, atol=0.0):
            raise ArithmeticError("Native and design fuel-mass accounting disagree")
        result["fresh_fuel_mass_check"] = native_mass
        result["fuel_mass_passed"] = mass_passed = bool(mass["passed"] and native_mass["passed"])
    result["feasible"] = bool(
        result["physics_passed"]
        and result["burnup_passed"]
        and result["bounds_passed"]
        and eol_passed
        and mass_passed
    )
    result["acceptable"] = bool(result["feasible"] and result["flatness_passed"])
    result["selection_merit"] = (
        result["metrics"]["power_variance"]
        if config.get("objective_kind") == "power_flatness"
        else result["merit"]
    )
    if "joint_objective" in config:
        reference = config["joint_objective"]
        changes = {
            "burnup_gain_MWd_tHM": result["metrics"]["discharge_burnup_MWd_tHM"]
            - reference["burnup_MWd_tHM"],
            "bol_reactivity_reduction_pcm": 1e5
            * (reference["bol_reactivity"] - result["bol_reactivity"]),
        }
        if reference.get("power_metric") == "maximum_channel_power_swing_MW":
            changes["maximum_channel_power_swing_reduction_MW"] = (
                reference["maximum_channel_power_swing_MW"]
                - result["metrics"]["maximum_channel_power_swing_MW"]
            )
        else:
            changes["temporal_variance_reduction"] = (
                reference["temporal_variance"] - result["metrics"]["power_variance"]
            )
        result["changes_from_reference"] = {
            **changes,
            "all_improved": all(v > 0 for v in changes.values()),
        }


def refresh_native_acceptance(record, config):
    """Reassess completed native candidates and select the best accepted.

    Parameters
    ----------
    record : dict
        Optimization record updated in place with native acceptance and best_native.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.
    """
    candidates = [(0, record["initial_native"])] + [
        (b["number"], b["native"]) for b in record["batches"] if "native" in b
    ]
    accepted = []
    for number, result in candidates:
        assess_native(result, config)
        if result["acceptable"]:
            accepted.append({"batch": number, **result})
    record["best_native"] = min(
        accepted, key=lambda result: result["selection_merit"], default=None
    )
    record["native_acceptance"] = {
        "objective_kind": config.get("objective_kind", "burnup_power"),
        "minimum_keff": config.get("native_minimum_keff", 1.0),
        "keff_standard_errors": config["native_keff_standard_errors"],
        "minimum_discharge_burnup_MWd_tHM": config.get("minimum_discharge_burnup_MWd_tHM"),
        "parameter_bounds": config["bounds"],
        "maximum_power_variance": config.get("maximum_native_power_variance"),
        "maximum_channel_temporal_rms": config.get("maximum_channel_temporal_rms"),
        "eol_keff": config.get("eol_keff"),
        "requirements": config["native_acceptance"],
    }
    if "fresh_fuel_mass" in config:
        record["native_acceptance"]["fresh_fuel_mass"] = config["fresh_fuel_mass"]


def native_summary(history, mapping, config, optimization):
    """Evaluate native residence objectives, precision and feasibility.

    Parameters
    ----------
    history : mapping
        Retained cycle schedules, native observations, design values and endpoint fuel
        states.
    mapping : FuelMap
        Region-to-physical-channel/axial mapping and symmetry weights.
    config : mapping
        Resolved physical cycle settings, objectives and sampling limits.
    optimization : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.

    Returns
    -------
    dict
        Native design metrics, physical discharge/power accounting, criticality and
        constraint acceptance.
    """
    keys = cycle_point_keys(history)
    observations = history["observations"]
    k = np.array([observations[key]["keff"] for key in keys])
    if k.shape != (len(keys), 2) or not np.isfinite(k).all() or (k[:, 1] < 0).any():
        raise ArithmeticError("Invalid native criticality observations")
    sampling = {
        "maximum_keff_standard_error": float(k[:, 1].max()),
        "maximum_power_relative_se_rms": max(
            observations[key]["power_relative_se_rms"] for key in keys
        ),
        "maximum_power_relative_se": max(
            observations[key]["power_relative_se_max"] for key in keys
        ),
    }
    sampling["passed"] = all(
        sampling[name] <= limit for name, limit in config["sampling_checks"].items()
    )
    discharge = discharge_summary(mapping, history["cycles"][-1]["end_state"])
    metrics = dict(history["cycles"][-1]["metrics"])
    profile = native_power_profile(
        history,
        mapping,
        include_temporal=(
            config["objective"].get("power_basis") == "channel_temporal"
            or config.get("record_temporal_power", False)
        ),
    )
    if config["objective"].get("power_basis") == "channel_temporal":
        if not np.isclose(
            metrics["power_variance"], profile["temporal"]["variance"], rtol=1e-12, atol=1e-14
        ):
            raise ArithmeticError("Native temporal profile differs from the cycle objective")
        metrics["power_variance"] = profile["temporal"]["variance"]
        metrics["maximum_channel_power_swing_MW"] = (
            profile["temporal"]["maximum_power_swing_W"] / 1e6
        )
    if not np.isclose(
        discharge["mean_burnup_MWd_tHM"], metrics["discharge_burnup_MWd_tHM"], rtol=1e-12
    ):
        raise ArithmeticError("Native burnup does not match the full-core discharge")
    if config["cycles"] == mapping.count and any(
        not row["id"].endswith("/axial/0") or not row["id"].startswith("cycle1/")
        for row in discharge["assemblies"]
    ):
        raise ArithmeticError("Discharged fuel did not traverse the complete three-cycle residence")
    peak = max(
        max(segment["burnup_MWd_tHM"])
        for cycle in history["cycles"]
        for segment in cycle["segments"]
    )
    energy_error = max(abs(cycle["energy_balance_relative"]) for cycle in history["cycles"])
    lower = k[:, 0] - optimization["native_keff_standard_errors"] * k[:, 1]
    peak_limit = config["objective"].get("peak_burnup_MWd_tHM")
    feasible = (
        float(lower.min()) >= optimization.get("native_minimum_keff", 1.0)
        and (peak_limit is None or peak <= peak_limit)
        and energy_error <= 1e-10
        and sampling["passed"]
    )
    bol_key = (
        bol_state_key(keys, optimization)
        if (optimization.get("objective_kind") == "bol_reactivity" or "joint_objective" in optimization)
        else None
    )
    score = merit(
        metrics["discharge_burnup_MWd_tHM"],
        metrics["power_variance"],
        config["objective"],
        optimization,
        bol_keff=observations[bol_key]["keff"][0] if bol_key else None,
        power_swing_MW=metrics.get("maximum_channel_power_swing_MW"),
    )
    metrics["loss"] = score
    result = {
        "design": history["design"],
        "metrics": metrics,
        "discharge": discharge,
        "power_profile": profile,
        "merit": score,
        "minimum_keff": float(k[:, 0].min()),
        "minimum_keff_minus_2se": float((k[:, 0] - 2 * k[:, 1]).min()),
        "minimum_keff_for_acceptance": float(lower.min()),
        "eol_keff_by_cycle": {
            str(c["cycle"]): observations[c["segments"][-1]["end"]]["keff"]
            for c in history["cycles"]
        },
        "limiting_state": keys[int(lower.argmin())],
        "maximum_peak_burnup_MWd_tHM": peak,
        "peak_burnup_limit_MWd_tHM": peak_limit,
        "peak_constraint_passed": peak_limit is None or peak <= peak_limit,
        "energy_balance_maximum_relative_error": energy_error,
        "sampling": sampling,
        "feasible": feasible,
        "keff_by_state": {key: observations[key]["keff"] for key in keys},
        "repeatability": history["cycles"][-1]["repeatability_to_previous"],
    }
    if config["objective"].get("power_basis") == "channel_temporal" or config.get(
        "record_temporal_power", False
    ):
        result["timing"] = dict(history["timing"])
    if "fresh_fuel_mass" in optimization:
        from optimization.fuel_constraints import native_fuel_mass_check

        if mapping.physical.base_case["design"] != history["design"]:
            raise ValueError("Native fuel mass belongs to another design")
        result["fresh_fuel_mass_check"] = native_fuel_mass_check(
            mapping.physical.base_case, optimization["fresh_fuel_mass"]
        )
    if bol_key:
        result.update(
            bol_state=bol_key,
            bol_keff=observations[bol_key]["keff"],
            bol_reactivity=1.0 - 1.0 / observations[bol_key]["keff"][0],
        )
    assess_native(result, optimization)
    return result


def backtrack_step(
    design, model, optimizer, config, pending, project_trial, progress, save_pending
):
    """Backtrack a saved Adam move using forward-only trial evaluations.

    Parameters
    ----------
    design : torch.nn.Module
        Live design parameters updated in place by optimization and projection.
    model : CycleObjective
        Connected residence objective storing its latest detached prediction.
    optimizer : torch.optim.Adam
        Live Adam optimizer whose moments and learning rates are part of restart state.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.
    pending : dict
        Restartable proposal with before/after optimizer states, predictions and accumulated
        trials.
    project_trial : callable
        Apply allowed parameter and geometry projections to a trial design.
    progress : callable
        Callback receiving keyword progress, timing and metric updates.
    save_pending : callable
        Persist the mutated pending-step checkpoint after each trial.

    Returns
    -------
    dict
        Chosen step record, including acceptance, fraction, tried predictions and any
        optimizer reset.
    """
    control = config["step_control"]
    original = pending["row"]
    before = pending["prediction_before"]
    origin, proposal = pending["design_before"], original["design"]
    trials = pending.setdefault(
        "trials",
        [
            {
                "fraction": 1.0,
                "design": deepcopy(proposal),
                "prediction": deepcopy(original["prediction"]),
                "wall_s": 0.0,
            }
        ],
    )
    optimizer.load_state_dict(pending["optimizer_after"])
    baseline_rank = prediction_rank(before, config)
    while prediction_rank(trials[-1]["prediction"], config) >= baseline_rank:
        if len(trials) - 1 >= control["maximum_backtracks"]:
            break
        fraction = trials[-1]["fraction"] / 2
        if len(trials) == 1:
            k_field = "eol_corrected_keff" if config.get("eol_keff") else "minimum_corrected_keff"
            k0 = before.get(k_field)
            k1 = original["prediction"].get(k_field)
            target = config.get("eol_keff", {}).get("target", config["surrogate_keff_target"])
            if k0 is not None and k1 is not None and (k0 - target) * (k1 - target) < 0:
                fraction = min(0.5, max(control["minimum_fraction"], (target - k0) / (k1 - k0)))
        if fraction < control["minimum_fraction"]:
            break
        set_parameters(
            design,
            {
                name: (
                    tensor(origin[name])
                    + fraction * (tensor(proposal[name]) - tensor(origin[name]))
                ).tolist()
                for name in origin
            },
        )
        project_trial(design)
        progress(
            status="surrogate_backtracking",
            step=original["step"],
            step_fraction=fraction,
            backtracking_evaluation=len(trials),
        )
        started = time.perf_counter()
        with torch.no_grad():
            model()
        trials.append(
            {
                "fraction": fraction,
                "design": parameters(design),
                "prediction": deepcopy(model.latest),
                "wall_s": time.perf_counter() - started,
            }
        )
        save_pending()
    accepted = prediction_rank(trials[-1]["prediction"], config) < baseline_rank
    chosen = trials[-1] if accepted else {"fraction": 0.0, "design": origin, "prediction": before}
    set_parameters(design, chosen["design"])
    model.latest = deepcopy(chosen["prediction"])
    if not accepted:
        optimizer.load_state_dict(pending["optimizer_before"])
    elif chosen["fraction"] < 1:
        # An oversized gradient near the constraint must not dominate the next Adam move.
        optimizer.state.clear()
        for group in optimizer.param_groups:
            group["lr"] *= chosen["fraction"]
    row = deepcopy(original)
    row.update(
        design=deepcopy(chosen["design"]),
        prediction=deepcopy(chosen["prediction"]),
        step_control={
            "accepted": accepted,
            "fraction": chosen["fraction"],
            "gradient_reused": True,
            "trials": deepcopy(trials),
            "optimizer_moments_reset": accepted and chosen["fraction"] < 1,
            "next_learning_rates": [g["lr"] for g in optimizer.param_groups],
        },
    )
    return row


@torch.no_grad()
def reset_projected_momentum(design, initial, optimizer, reproject, *, anchor_feasible):
    """Discard projected uphill Adam momentum at a feasible anchor.

    Parameters
    ----------
    design : torch.nn.Module
        Live design parameters updated in place by optimization and projection.
    initial : mapping
        Parameter values before the Adam step.
    optimizer : torch.optim.Adam
        Live Adam optimizer whose moments and learning rates are part of restart state.
    reproject : callable
        Reapply the same constraints after taking a step with reset moments.
    anchor_feasible : bool
        Whether the origin meets the prediction's feasibility ranking.

    Returns
    -------
    dict
        Before/after directional derivatives and whether optimizer moments were reset; no
        model solve is added.
    """

    def slope():
        """Contract the saved gradient with the currently projected move.

        Returns
        -------
        float
            Objective directional derivative along the parameter displacement.
        """
        return sum(
            float((p.grad * (p - tensor(initial[name]))).sum())
            for name, p in design.named_parameters()
            if p.grad is not None
        )

    before = slope()
    reset = bool(anchor_feasible and before >= 0.0)
    result = {
        "anchor_feasible": bool(anchor_feasible),
        "directional_derivative_before": before,
        "optimizer_moments_reset": reset,
        "gradient_reused": True,
    }
    if reset:
        result["discarded_projected_design"] = parameters(design)
        for name, p in design.named_parameters():
            p.copy_(tensor(initial[name]))
        optimizer.state.clear()
        optimizer.step()
        reproject()
    result["directional_derivative_after"] = slope()
    return result


def optimize_batch(
    root, shared, design, config, batch, checkpoint, checkpoint_path, progress, save_record
):
    """Run or resume a bounded batch with fixed fitted coefficients.

    Parameters
    ----------
    root : CycleCase
        Resolved case with model factories, input settings and result paths.
    shared : SharedCycle
        Connected residence model with fixed shared correction coefficients and live design
        parameters.
    design : torch.nn.Module
        Live design parameters updated in place by optimization and projection.
    config : mapping
        Resolved optimization objective, bounds, native acceptance and step-control
        settings.
    batch : dict
        Batch record updated with every evaluated proposal and the selected design.
    checkpoint : dict
        Restart state updated with live/best design and Adam state.
    checkpoint_path : str or pathlib.Path
        Destination for the optimizer restart checkpoint.
    progress : callable
        Callback receiving keyword progress, timing and metric updates.
    save_record : callable
        Persist the enclosing optimization result record.

    Returns
    -------
    dict
        Best selected design parameter values; the live design is restored to these values.
    """
    model = CycleObjective(shared, config)
    if config.get("surrogate_keff_policy") == "anchor_corrected_soft":
        if "keff_correction" not in batch:
            fit_path = root.result("fit")
            fitted = json.loads(fit_path.read_text())
            batch["keff_correction"] = keff_correction(
                shared.history["observations"], fitted["selected"]["coupled"]["predictions"], config
            )
            batch["keff_correction"]["calibration_sha256"] = sha(fit_path)
        model.keff_correction = batch["keff_correction"]
    optimizer = torch.optim.Adam(
        [
            {"params": [p], "lr": config["learning_rates"][name]}
            for name, p in design.named_parameters()
        ]
    )
    use_eol_projection = config.get("step_control", {}).get("project_eol", False)

    def prepare_projection(*, reuse_as_anchor):
        """Compute and retain the EOL constraint linearization.

        Parameters
        ----------
        reuse_as_anchor : bool
            Record whether this forward evaluation also supplies the batch's anchor prediction.

        Returns
        -------
        dict
            Saved EOL derivative and timing record.
        """
        progress(
            status="differentiating_eol_constraint",
            completed_step=checkpoint.get("completed_steps", 0),
        )
        linearization = linearize_eol(model, design)
        linearization["reused_as_anchor"] = reuse_as_anchor
        checkpoint["eol_linearization"] = linearization
        batch["eol_linearization"] = deepcopy(linearization)
        return linearization

    if "current_optimizer" in checkpoint:
        set_parameters(design, checkpoint["current_design"])
        optimizer.load_state_dict(checkpoint["current_optimizer"])
    else:
        if use_eol_projection:
            linearization = prepare_projection(reuse_as_anchor=True)
            batch["anchor_forward_wall_s"] = linearization["forward_wall_s"]
        else:
            anchor_started = time.perf_counter()
            with torch.no_grad():
                model()
            batch["anchor_forward_wall_s"] = time.perf_counter() - anchor_started
        batch["prediction_before"] = deepcopy(model.latest)
        checkpoint.update(
            current_design=parameters(design),
            current_optimizer=deepcopy(optimizer.state_dict()),
            best_design=parameters(design),
            best_optimizer=deepcopy(optimizer.state_dict()),
            best_prediction=deepcopy(model.latest),
            selected_step=0,
            completed_steps=0,
        )
        save_checkpoint(checkpoint_path, checkpoint)
        save_record()
    if use_eol_projection and "eol_linearization" not in checkpoint:
        prepare_projection(reuse_as_anchor=False)
        save_checkpoint(checkpoint_path, checkpoint)
        save_record()

    def finish_step(row):
        """Retain a completed trial and update the best batch checkpoint.

        Parameters
        ----------
        row : dict
            Completed surrogate proposal and step-control diagnostics.
        """
        step = row["step"]
        if batch["steps"] and batch["steps"][-1]["step"] == step:
            batch["steps"][-1] = row
        else:
            batch["steps"].append(row)
        checkpoint.update(
            current_design=parameters(design),
            current_optimizer=deepcopy(optimizer.state_dict()),
            completed_steps=step,
        )
        predicted = row["prediction"]
        if prediction_rank(predicted, config) < prediction_rank(
            checkpoint["best_prediction"], config
        ):
            checkpoint.update(
                best_design=parameters(design),
                best_optimizer=deepcopy(optimizer.state_dict()),
                best_prediction=deepcopy(predicted),
                selected_step=step,
            )
        checkpoint.pop("pending_step", None)
        save_checkpoint(checkpoint_path, checkpoint)
        save_record()
        progress(
            completed_step=step,
            predicted_metrics=predicted["metrics"],
            minimum_keff=predicted["minimum_keff"],
            minimum_corrected_keff=predicted.get("minimum_corrected_keff"),
            eol_corrected_keff=predicted.get("eol_corrected_keff"),
            eol_target_check=predicted.get("eol_target_check"),
        )

    def finish_pending():
        """Resume backtracking and checkpoint its final decision.

        Returns
        -------
        bool
            Whether the pending proposal produced an accepted improvement.
        """
        row = backtrack_step(
            design,
            model,
            optimizer,
            config,
            checkpoint["pending_step"],
            lambda d: project(d, batch["anchor_design"], config, root),
            progress,
            lambda: save_checkpoint(checkpoint_path, checkpoint),
        )
        finish_step(row)
        return row["step_control"]["accepted"]

    continue_steps = finish_pending() if "pending_step" in checkpoint else True
    for step in range(checkpoint["completed_steps"] + 1, config["steps_per_refresh"] + 1):
        if not continue_steps:
            break
        progress(status="surrogate_optimization", step=step, steps=config["steps_per_refresh"])
        before = {
            "design_before": parameters(design),
            "optimizer_before": deepcopy(optimizer.state_dict()),
            "prediction_before": deepcopy(
                batch["steps"][-1]["prediction"] if batch["steps"] else batch["prediction_before"]
            ),
        }
        projection = {}

        def project_candidate(d, initial):
            """Apply proposal constraints and optional momentum correction.

            Parameters
            ----------
            d : torch.nn.Module
                Live design parameters updated in place by optimization and projection.
            initial : mapping
                Parameter values before the current optimizer update.
            """
            def project_once():
                """Apply trust/geometry bounds and the optional EOL projection."""
                project(d, batch["anchor_design"], config, root)
                if use_eol_projection:
                    projection.update(
                        project_eol_step(
                            d,
                            initial,
                            before["prediction_before"],
                            batch["anchor_design"],
                            config,
                            optimizer,
                            checkpoint["eol_linearization"],
                            root,
                        )
                    )

            project_once()
            if config.get("step_control", {}).get("reset_projected_non_descent", False):
                projection["descent_control"] = reset_projected_momentum(
                    d,
                    initial,
                    optimizer,
                    project_once,
                    anchor_feasible=prediction_rank(before["prediction_before"], config)[0] == 0,
                )

        result = minimize(
            design,
            model,
            lambda output: output["loss"],
            optimizer=optimizer,
            steps=1,
            project=project_candidate,
        )
        predicted = deepcopy(model.latest)
        row = {
            "step": step,
            "design": parameters(design),
            "prediction": predicted,
            "gradient": result["gradient"],
            "objective_before": result["objective_before"],
            "forward_wall_s": result["forward_wall_s"],
            "backward_wall_s": result["backward_wall_s"],
        }
        if projection:
            row["eol_projection"] = projection
        if config.get("step_control"):
            checkpoint["pending_step"] = {
                **before,
                "row": row,
                "optimizer_after": deepcopy(optimizer.state_dict()),
            }
            save_checkpoint(checkpoint_path, checkpoint)
            continue_steps = finish_pending()
        else:
            finish_step(row)
    set_parameters(design, checkpoint["best_design"])
    batch.update(
        proposal=checkpoint["best_design"],
        predicted=checkpoint["best_prediction"],
        selected_step=checkpoint["selected_step"],
    )
    checkpoint["phase"] = "native" if checkpoint["selected_step"] else "no_improving_step"
    save_checkpoint(checkpoint_path, checkpoint)
    save_record()
    return batch["proposal"]


def refit_selected(root, physical, chain, mapping, entry, history, initializer, progress):
    """Refit the selected correction basis to refreshed native evidence.

    Parameters
    ----------
    root : CycleCase
        Resolved case with model factories, input settings and result paths.
    physical : IsotopicInventory
        Full-chain inventory binding and native case factory.
    chain : SparseChain
        Differentiable full-chain CRAM propagator using per-parent rates in s⁻¹.
    mapping : FuelMap
        Region-to-physical-channel/axial mapping and symmetry weights.
    entry : FuelState
        Fuel inventory in atoms/cm³, energy in J, initial heavy-metal mass in kg and
        resident identities.
    history : mapping
        Retained cycle schedules, native observations, design values and endpoint fuel
        states.
    initializer : mapping
        Previously selected layout, basis names, log factors and calibration identity.
    progress : callable
        Callback receiving keyword progress, timing and metric updates.

    Returns
    -------
    dict
        Matching saved or newly fitted pooled/connected calibration record.
    """
    from coupling.calibration import SharedCycle, calibration_identity, fit_shared

    workflow = root
    path = workflow.result("fit")
    identity = calibration_identity(workflow)
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get("status") in ("checked", "exploratory") and all(
        previous.get(k) == v for k, v in identity.items()
    ):
        return previous
    design = workflow.design()
    set_parameters(design, history["design"])
    design.requires_grad_(False)
    model = SharedCycle(
        workflow, physical, chain, mapping, entry, history, design, initializer["layout"]
    )
    if list(model.parameter_names) != initializer["parameter_names"]:
        raise ValueError("Native refresh changed the selected correction basis")
    model.coefficients.initialize(initializer["log_factors"])
    started = time.perf_counter()
    result = {
        "status": "calibrating",
        **identity,
        "scope": model.fit_config["scope"],
        "selection_policy": "Reuse the selected compact basis and refit it to every new native history.",
        "initializer_calibration_sha256": initializer["calibration_sha256"],
    }
    candidate = {"layout": model.layout, "parameter_names": list(model.parameter_names)}
    matching = all(previous.get(k) == v for k, v in identity.items())
    if matching and previous.get("selected", {}).get("pooled"):
        candidate["pooled"] = previous["selected"]["pooled"]
        model.coefficients.initialize(candidate["pooled"]["log_factors"])
    else:
        candidate["pooled"] = fit_shared(model, coupled=False, progress=progress)
    result.update(
        selected=candidate, candidate_bases=[candidate], wall_s=time.perf_counter() - started
    )
    save_json(path, result)
    candidate["coupled"] = fit_shared(model, coupled=True, progress=progress)
    result.update(
        field_quality_passed=candidate["coupled"]["accepted"],
        status="checked" if candidate["coupled"]["accepted"] else "exploratory",
        wall_s=time.perf_counter() - started,
    )
    save_json(path, result)
    return result


@interrupt_on_sigterm()
def refresh_native(workflow):
    """Complete one restartable surrogate, native and refit batch.

    Parameters
    ----------
    workflow : CycleCase
        Resolved case with model factories, input settings and result paths.

    Notes
    -----
    The function writes optimizer checkpoints, native histories, calibrated coefficients and
    progress records. A failed or non-improving proposal is recorded without claiming a new
    accepted native design.
    """
    from runtime.processes import native_library_path
    import torch

    os.environ["OPENMC_REFERENCE_LIBRARY"] = str(native_library_path())
    from coupling.calibration import calibration_identity, restore_shared
    from fuel_cycle.residence import state_payload
    from runtime.autograd import GradientStorage
    from coupling.history import native_history, restore_baseline, retain_points
    from runtime.io import save_state

    torch.set_num_threads(1)
    workflow.prepare_outputs()
    record_path, status_path = (workflow.result("optimization"), workflow.result("status"))
    config = workflow.settings()
    optimization = workflow.settings("optimization")
    work_root = workflow.work / "optimization"
    checkpoint_key = optimization.get("checkpoint_key", optimization.get("objective_kind"))
    work = work_root / checkpoint_key if checkpoint_key else work_root
    checkpoint_path = work / "optimizer.pt"
    started = time.perf_counter()
    status = {
        "status": "prepared",
        "pid": os.getpid(),
        "native_pid": None,
        "scope": optimization["scope"],
        "case_name": workflow.name,
        "settings": config,
        "optimization_settings": optimization,
    }

    def progress(**changes):
        """Persist search progress and release unused native-stage memory.

        Parameters
        ----------
        **changes : object
            Named status, timing, stage or diagnostic updates.
        """
        if changes.get("status") == "native_history":
            status.update(segment=0, elapsed_cycle_days=0.0)
        status.update(changes)
        status["wall_s"] = time.perf_counter() - started
        save_json(status_path, status)
        if changes.get("status") == "native_completed":
            GradientStorage.release_unused()

    record = json.loads(record_path.read_text()) if record_path.is_file() else None
    if (
        record is not None
        and optimization.get("search_id")
        and (record.get("search_id") != optimization["search_id"])
    ):
        record = None
    checkpoint = (
        torch.load(checkpoint_path, weights_only=True) if checkpoint_path.is_file() else None
    )
    if checkpoint is not None and checkpoint["phase"] == "finished":
        checkpoint = None
    try:
        if record is not None and (
            record.get("schema_version") != 1 or record.get("case_definition") != workflow.definition()
        ):
            raise ValueError("Existing optimization does not match the current physical case")
        if checkpoint is not None:
            if (
                checkpoint.get("settings") != config
                or checkpoint.get("optimization_settings") != optimization
            ):
                raise ValueError(
                    "Interrupted batch uses different settings; reconcile before resuming"
                )
            if record is None or checkpoint["batch_number"] != len(record["batches"]):
                raise ValueError("Optimization checkpoint and result record do not match")
            batch = checkpoint.get("batch", record["batches"][-1])
            record["batches"][-1] = batch
        else:
            physical, chain, mapping, entry, history = restore_baseline(workflow)
            fit_path = workflow.result("fit")
            fitted = json.loads(fit_path.read_text())
            identity = calibration_identity(workflow)
            if any((fitted.get(k) != v for k, v in identity.items())):
                raise ValueError(
                    "Refresh the current calibration before starting another design batch"
                )
            if record is None:
                check_path = workflow.result("gradient_checks")
                checks = json.loads(check_path.read_text())
                if (
                    checks["status"] not in ("checked", "gradients_computed")
                    or checks.get("settings") != config
                    or checks["calibration_sha256"] != sha(fit_path)
                ):
                    raise ValueError("Compute the current full-residence autograd result first")
                if checks["status"] == "gradients_computed" and (
                    not checks.get("gradient_graph_passed")
                ):
                    raise ValueError("The full-residence gradient graph is incomplete")
                record = {
                    "schema_version": 1,
                    "status": "running",
                    "objective": optimization["objective"],
                    "search_id": optimization.get("search_id"),
                    "case_name": workflow.name,
                    "settings": config,
                    "gradient_checks_sha256": sha(check_path),
                    "case_definition": identity["case_definition"],
                    "initial_native": native_summary(history, mapping, config, optimization),
                    "batches": [],
                }
                record["initial_calibration"] = {
                    "sha256": sha(fit_path),
                    "wall_s": fitted["wall_s"],
                    "coefficient_reuse": fitted.get("coefficient_reuse"),
                }
            batch = {
                "number": len(record["batches"]) + 1,
                "status": "surrogate_optimization",
                "anchor_design": history["design"],
                "anchor_calibration_sha256": sha(fit_path),
                "settings": optimization,
                "steps": [],
            }
            record["batches"].append(batch)
            selected = fitted["selected"]
            checkpoint = {
                "phase": "surrogate",
                "batch_number": batch["number"],
                "settings": config,
                "optimization_settings": optimization,
                "initializer": {
                    "layout": selected["layout"],
                    "parameter_names": selected["parameter_names"],
                    "log_factors": selected["coupled"]["log_factors"],
                    "calibration_sha256": sha(fit_path),
                },
            }
            del physical, chain, mapping, entry, history
        record.update(
            status="running", objective=optimization["objective"], active_settings=optimization
        )

        def save_record():
            """Reassess native candidates and save the optimization record."""
            refresh_native_acceptance(record, optimization)
            save_json(record_path, record)

        def checkpoint_phase(phase):
            """Save the current batch phase with its complete result record.

            Parameters
            ----------
            phase : str
                Next restart phase, such as surrogate, native, refit or finished.
            """
            checkpoint["phase"] = phase
            checkpoint["batch"] = batch
            save_checkpoint(checkpoint_path, checkpoint)
            save_record()

        progress(batch=batch["number"], phase=checkpoint["phase"])
        checkpoint_phase(checkpoint["phase"])
        if checkpoint["phase"] == "surrogate":
            physical, chain, mapping, entry, history = restore_baseline(workflow)
            design = workflow.design()
            set_parameters(design, history["design"])
            shared, fitted = restore_shared(
                workflow, physical, chain, mapping, entry, history, design
            )
            if sha(workflow.result("fit")) != batch["anchor_calibration_sha256"]:
                raise ValueError("A fixed-fit optimization batch cannot change calibration midway")
            optimize_batch(
                workflow,
                shared,
                design,
                optimization,
                batch,
                checkpoint,
                checkpoint_path,
                progress,
                save_record,
            )
            del shared, fitted, physical, chain, mapping, entry, history, design
            GradientStorage.release_unused()
        if checkpoint["phase"] == "no_improving_step":
            batch["status"] = "no_improving_surrogate_step"
            record["status"] = "local_step_review_needed"
            checkpoint_phase("finished")
            progress(
                status=record["status"],
                pid=None,
                native_pid=None,
                next_action="Review the projected gradients and step sizes; no new native design was claimed.",
            )
            return
        if checkpoint["phase"] == "native":
            GradientStorage.release_unused()
            batch["status"] = "native_verification"
            checkpoint_phase("native")
            progress(
                status="native_verification",
                phase="native",
                design=batch["proposal"],
                predicted_metrics=batch["predicted"]["metrics"],
                minimum_keff=batch["predicted"]["minimum_keff"],
                minimum_corrected_keff=batch["predicted"].get("minimum_corrected_keff"),
                eol_corrected_keff=batch["predicted"].get("eol_corrected_keff"),
                eol_target_check=batch["predicted"].get("eol_target_check"),
                selected_step=batch["selected_step"],
            )
            physical, chain, mapping, entry, history = native_history(
                workflow,
                work / "native",
                config,
                progress,
                design_values=batch["proposal"],
                initialize=False,
            )
            batch["native"] = native_summary(history, mapping, config, optimization)
            prediction = batch["predicted"]
            batch["prediction_error"] = {
                "discharge_burnup_relative": prediction["metrics"]["discharge_burnup_MWd_tHM"]
                / batch["native"]["metrics"]["discharge_burnup_MWd_tHM"]
                - 1,
                "power_variance_absolute": prediction["metrics"]["power_variance"]
                - batch["native"]["metrics"]["power_variance"],
                "keff_maximum_absolute": max(
                    (
                        abs(prediction["keff_by_state"][key] - value[0])
                        for key, value in batch["native"]["keff_by_state"].items()
                    )
                ),
            }
            if "corrected_keff_by_state" in prediction:
                batch["prediction_error"]["corrected_keff_maximum_absolute"] = max(
                    (
                        abs(prediction["corrected_keff_by_state"][key] - value[0])
                        for key, value in batch["native"]["keff_by_state"].items()
                    )
                )
            save_state(workflow.result("history"), history)
            save_state(
                workflow.result("entry"), {"design": history["design"], **state_payload(entry)}
            )
            retain_points(workflow, work / "native", history)
            checkpoint_phase("refit")
            del physical, chain, mapping, entry, history
            GradientStorage.release_unused()
        if checkpoint["phase"] == "refit":
            progress(status="calibrating_shared_cycle", phase="refit", native_pid=None)
            physical, chain, mapping, entry, history = restore_baseline(workflow)
            fit = refit_selected(
                workflow,
                physical,
                chain,
                mapping,
                entry,
                history,
                checkpoint["initializer"],
                progress,
            )
            batch["refit"] = {
                "calibration_sha256": sha(workflow.result("fit")),
                "status": fit["status"],
                "parameters": len(fit["selected"]["parameter_names"]),
                "wall_s": fit["wall_s"],
                "errors": fit["selected"]["coupled"]["errors"],
            }
            batch["status"] = "native_refreshed"
            record["status"] = "native_refresh_completed"
            checkpoint_phase("finished")
            progress(
                status=record["status"],
                phase="idle",
                pid=None,
                native_pid=None,
                native_metrics=batch["native"]["metrics"],
                native_feasible=batch["native"]["feasible"],
                native_acceptable=batch["native"]["acceptable"],
                native_minimum_keff=batch["native"]["minimum_keff"],
                native_minimum_keff_minus_2se=batch["native"]["minimum_keff_minus_2se"],
                next_action="Review native improvement and continue batches with the refreshed shared model.",
            )
    except BaseException as error:
        if record is not None:
            record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            if "batch" in locals():
                batch["status"] = record["status"]
            save_json(record_path, record)
        progress(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            pid=None,
            native_pid=None,
            error=f"{type(error).__name__}: {error}",
        )
        raise


def run_search(workflow):
    """Continue native refreshes to a declared budget or stall limit.

    Parameters
    ----------
    workflow : CycleCase
        Resolved case with model factories, input settings and result paths.

    Notes
    -----
    Search status and the stopping reason are saved to case result paths. Completion
    establishes that the configured stopping rule was reached, not global optimality.
    """
    context = workflow
    record_path, status_path = (context.result("optimization"), context.result("status"))
    options = context.settings("optimization")
    maximum = options.get("maximum_refreshes", 1)
    previous_best, stalls = (None, 0)
    while True:
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            if record.get("search_id") == options.get("search_id"):
                if record.get("status") in ("search_stopped", "local_step_review_needed"):
                    return
                completed = [b for b in record["batches"] if b.get("status") == "native_refreshed"]
                if len(completed) >= maximum:
                    reason = "declared_refresh_limit"
                    break
                best = record.get("best_native")
                previous_best = best["selection_merit"] if best else None
                stalls = record.get("search_progress", {}).get("stalled_refreshes", 0)
        refresh_native(context)
        record = json.loads(record_path.read_text())
        if record["status"] == "local_step_review_needed":
            reason = "no_improving_surrogate_step"
            break
        latest = record["batches"][-1].get("native")
        best = record.get("best_native")
        if (
            latest is not None
            and latest["acceptable"]
            and (best is not None)
            and (previous_best is not None)
        ):
            improvement = previous_best - best["selection_merit"]
            stalls = (
                stalls + 1
                if improvement < options.get("native_score_stall_tolerance", 0.002)
                else 0
            )
        else:
            stalls = 0
        record["search_progress"] = {"stalled_refreshes": stalls, "maximum_refreshes": maximum}
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        if stalls >= options.get("native_stall_batches", 2):
            reason = "native_score_stalled"
            break
    record["status"] = (
        "search_limit_reached" if reason == "declared_refresh_limit" else "search_stopped"
    )
    record["search_progress"] = {
        "stopping_reason": reason,
        "stalled_refreshes": stalls,
        "maximum_refreshes": maximum,
        "global_optimality_claimed": False,
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    status = json.loads(status_path.read_text())
    status.update(
        status=record["status"],
        phase="idle",
        pid=None,
        native_pid=None,
        stopping_reason=reason,
        next_action="Compare the initial and best feasible native designs.",
    )
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(record["search_progress"]), flush=True)
