"""Search, derivative and native-comparison figures independent of case parameters."""

import math
import numpy as np
import matplotlib.pyplot as plt

from plotting.history import finish


def native_cases(record):
    """Collect the initial and completed batch-native evaluations.

    Parameters
    ----------
    record : mapping
        Optimization record containing the initial native evaluation and subsequent proposal
        batches.

    Returns
    -------
    list of tuple
        Pairs of batch number and native evaluation record, starting with batch zero.
    """
    return [(0, record["initial_native"])] + [
        (batch["number"], batch["native"]) for batch in record["batches"] if "native" in batch]


def plot_gradients(gradient, path, *, title="Objective derivatives", labels=None):
    """Plot objective derivatives in each parameter's native units.

    Parameters
    ----------
    gradient : mapping
        Parameter names mapped to objective derivative arrays.
    path : str or pathlib.Path
        Destination figure file; missing parent directories are created.
    title : str, optional
        Figure heading.
    labels : mapping or None, optional
        Optional parameter-to-component labels; absent labels use component indices.
    """
    labels = {} if labels is None else labels
    rows = math.ceil(len(gradient) / 2)
    fig, panels = plt.subplots(rows, 2, figsize=(11, 3 * rows), layout="constrained", squeeze=False)
    for axis, (name, values) in zip(panels.flat, gradient.items()):
        values = np.asarray(values).reshape(-1)
        names = labels.get(name, [str(i) for i in range(len(values))])
        axis.bar(np.arange(len(values)), values, color="#31738b")
        axis.set(title=name.replace("_", " "), xticks=np.arange(len(values)), xticklabels=names,
                 ylabel="Loss derivative / parameter unit")
        axis.tick_params(axis="x", rotation=60 if len(values) > 8 else 0, labelsize=7)
        axis.axhline(0., color=".5", lw=.7)
    for axis in list(panels.flat)[len(gradient):]:
        axis.set_visible(False)
    fig.suptitle(title)
    finish(fig, path)


def plot_search(record, options, path, *, title="Optimization progress"):
    """Plot surrogate proposals alongside native acceptance decisions.

    Parameters
    ----------
    record : mapping
        Optimization record containing the initial native evaluation and subsequent proposal
        batches.
    options : mapping
        Current objective and constraint configuration used to choose plotted metrics and
        limits.
    path : str or pathlib.Path
        Destination figure file; missing parent directories are created.
    title : str, optional
        Figure heading.
    """
    objectives = [("Discharge burnup (MWd/tHM)", lambda result: result["metrics"]["discharge_burnup_MWd_tHM"])]
    if "fresh_fuel_mass" in options:
        from optimization.fuel_constraints import fuel_mass_check
        objectives.append(("Fresh fuel mass (kg)", lambda result: fuel_mass_check(
            result["design"], options["fresh_fuel_mass"])["fuel_mass_kg"]))
    elif options.get("joint_objective", {}).get("power_metric") == "maximum_channel_power_swing_MW":
        def swing(result):
            """Extract the largest physical channel-power swing.

            Parameters
            ----------
            result : mapping
                Native evaluation with a power profile or surrogate result with swing metrics.

            Returns
            -------
            float
                Largest sampled channel maximum-minus-minimum power in MW.
            """
            if "power_profile" in result:
                power = result["power_profile"]["temporal"]["channel_power_W"]
                return np.ptp(power, axis=0).max() / 1e6
            return result["metrics"]["maximum_channel_power_swing_MW"]
        objectives.append(("Largest channel power swing (MW)", swing))
    else:
        objectives.append(("Power RMS (%)", lambda result: 100 * np.sqrt(result["metrics"]["power_variance"])))
    if "eol_keff" in options:
        cycle = str(options["eol_keff"]["cycle"])
        def keff(result):
            """Extract the configured end-of-life multiplication factor.

            Parameters
            ----------
            result : mapping
                Native or surrogate evaluation containing cycle-end criticality information.

            Returns
            -------
            float
                Native nominal or corrected surrogate EOL effective multiplication factor.
            """
            if "power_profile" in result:
                return result["eol_keff_by_cycle"][cycle][0]
            if "eol_corrected_keff" in result:
                return result["eol_corrected_keff"]
            from coupling.optimization import eol_state_key
            values = result.get("corrected_keff_by_state", result["keff_by_state"])
            return values[eol_state_key(values, options)]
        objectives.append(("EOL effective multiplication factor", keff))
    else:
        objectives.append(("Minimum sampled effective multiplication factor", lambda result: result.get(
            "minimum_corrected_keff", result["minimum_keff"])))
    if "bol_reactivity" in options:
        objectives.append(("BOL reactivity", lambda result: 1. - 1. / (
            result["bol_keff"][0] if "power_profile" in result else result["bol_corrected_keff"])))
    rows = math.ceil((len(objectives) + 1) / 2)
    fig, panels = plt.subplots(rows, 2, figsize=(12, 3.5 * rows), layout="constrained", squeeze=False)
    axes = list(panels.flat)
    offset, proposals = 0, []
    native = [(0, record["initial_native"])]
    for batch in record["batches"]:
        points = [(offset + row["step"], {**row["prediction"], "design": row["design"]})
                  for row in batch["steps"]]
        previous = batch.get("settings", {})
        current = (previous.get("objective_kind") == options.get("objective_kind")
                   and previous.get("eol_keff") == options.get("eol_keff"))
        for axis, (_, metric) in zip(axes, objectives):
            available = points
            if available:
                axis.plot([x for x, _ in available], [metric(result) for _, result in available],
                          ".--", color="#31738b" if current else "#a4aaae", alpha=.8)
        proposals.extend(points)
        if "native" in batch:
            native.append((offset + batch.get("selected_step", len(batch["steps"])), batch["native"]))
        offset += max((row["step"] for row in batch["steps"]), default=0)
    for axis, (ylabel, metric) in zip(axes, objectives):
        for i, (x, result) in enumerate(native):
            accepted = result.get("acceptable", result["feasible"])
            axis.scatter(x, metric(result), marker="s", s=38, zorder=4,
                         color="#437b51" if accepted else "#b75f21")
        axis.set(xlabel="Surrogate proposal", ylabel=ylabel)
        axis.grid(alpha=.15)
    if "fresh_fuel_mass" in options:
        axes[1].axhline(options["fresh_fuel_mass"]["maximum_fresh_fuel_mass_kg"], color=".4", ls=":")
    if "eol_keff" in options:
        axes[2].axhline(options["eol_keff"]["target"], color=".4", ls=":")
    parameters = axes[len(objectives)]
    origin = record["initial_native"]["design"]
    for index, (name, initial) in enumerate(origin.items()):
        initial = np.asarray(initial, dtype=float)
        if proposals:
            values = np.array([result["design"][name] for _, result in proposals])
            scale = np.where(abs(initial) > 1e-15, abs(initial), 1.)
            change = (values - initial) / scale * 100
            for column in range(change.shape[-1]):
                parameters.plot([x for x, _ in proposals], change[:, column], lw=.8,
                                color=f"C{index % 10}",
                                label=name.replace("_", " ") if column == 0 else None)
    parameters.set(xlabel="Surrogate proposal", ylabel="Change from initial value (%)")
    if proposals:
        parameters.legend(fontsize=7, ncol=2)
    for axis in axes[len(objectives) + 1:]:
        axis.set_visible(False)
    fig.suptitle(title + "\nBlue: surrogate; green: accepted native; orange: native constraints unmet")
    finish(fig, path)


def plot_native_comparison(record, path, *, title="Native comparison", labels=None):
    """Compare initial and latest native channel power and discharge.

    Parameters
    ----------
    record : mapping
        Optimization record containing the initial native evaluation and subsequent proposal
        batches.
    path : str or pathlib.Path
        Destination figure file; missing parent directories are created.
    title : str, optional
        Figure heading.
    labels : sequence of str or None, optional
        Display names in the native physical-channel order.

    Raises
    ------
    ValueError
        The evaluations use different physical-channel identities.
    """
    initial, latest = record["initial_native"], native_cases(record)[-1][1]
    channels = initial["power_profile"]["channel_ids"]
    labels = channels if labels is None else labels
    if channels != latest["power_profile"]["channel_ids"]:
        raise ValueError("Native comparisons require matching physical channel identities")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    positions = np.arange(len(channels))
    for offset, result, label, color in ((-.18, initial, "Initial", "#a4aaae"),
                                        (.18, latest, "Latest", "#31738b")):
        axes[0].bar(positions + offset, np.asarray(result["power_profile"]["channel_mean_power_W"]) / 1e6,
                    width=.35, label=label, color=color)
        discharge = result["discharge"]
        values = {row["channel"]: row["burnup_MWd_tHM"] for row in discharge["assemblies"]}
        axes[1].bar(positions + offset, [values[channel] for channel in channels], width=.35, label=label, color=color)
    for axis, ylabel in zip(axes, ("Mean channel power (MW)", "Discharge burnup (MWd/tHM)")):
        axis.set(xticks=positions, xticklabels=labels, ylabel=ylabel)
        axis.tick_params(axis="x", rotation=45, labelsize=8)
        axis.legend(fontsize=8)
    fig.suptitle(title)
    finish(fig, path)


def plot_temporal_comparison(record, path, *, title="Channel power histories", labels=None):
    """Compare initial and latest native channel-power trajectories.

    Parameters
    ----------
    record : mapping
        Optimization record containing the initial native evaluation and subsequent proposal
        batches.
    path : str or pathlib.Path
        Destination figure file; missing parent directories are created.
    title : str, optional
        Figure heading.
    labels : sequence of str or None, optional
        Display names in the native physical-channel order.

    Raises
    ------
    ValueError
        The evaluations use different physical-channel identities.

    Notes
    -----
    No figure is written when either evaluation lacks a temporal power profile.
    """
    initial, latest = record["initial_native"], native_cases(record)[-1][1]
    channels = initial["power_profile"]["channel_ids"]
    labels = channels if labels is None else labels
    if channels != latest["power_profile"]["channel_ids"]:
        raise ValueError("Temporal comparisons require matching physical channel identities")
    if any("temporal" not in result["power_profile"] for result in (initial, latest)):
        return
    rows = math.ceil(len(channels) / 3)
    fig, panels = plt.subplots(rows, 3, figsize=(12, 3 * rows), layout="constrained", squeeze=False)
    for index, axis in enumerate(list(panels.flat)[:len(channels)]):
        for result, label, color in ((initial, "Initial", "#a4aaae"), (latest, "Latest", "#31738b")):
            temporal = result["power_profile"]["temporal"]
            axis.plot(temporal["time_days"], np.asarray(temporal["channel_power_W"])[:, index] / 1e6,
                      ".-", label=label, color=color)
        axis.set(title=labels[index], xlabel="Time within cycle (days)", ylabel="Channel power (MW)")
        axis.legend(fontsize=7)
    for axis in list(panels.flat)[len(channels):]:
        axis.set_visible(False)
    fig.suptitle(title)
    finish(fig, path)
