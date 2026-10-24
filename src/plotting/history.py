"""Plots of native histories and fitted fields with caller-supplied labels."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sample_order(history):
    """Order saved cycle samples along a continuous time axis.

    Parameters
    ----------
    history : mapping
        Saved native cycle history containing observations and beginning/midpoint/end state
        keys.

    Returns
    -------
    keys : list of str
        Beginning, midpoint and end state keys in cycle order.
    days : numpy.ndarray
        Absolute sample times in days, including repeated cycle boundaries.
    """
    keys, days, offset = [], [], 0.
    for cycle in history["cycles"]:
        keys.append(cycle["bol"])
        days.append(offset)
        for segment in cycle["segments"]:
            keys.extend((segment["mid"], segment["end"]))
            days.extend((offset + segment["time_days"] - segment["duration_days"] / 2,
                         offset + segment["time_days"]))
        offset += cycle["cycle_days"]
    return keys, np.array(days)


def finish(fig, path, **savefig_options):
    """Save a figure and close its Matplotlib resources.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Completed figure to write.
    path : str or pathlib.Path
        Destination figure file; missing parent directories are created.
    **savefig_options : object
        Additional keyword arguments forwarded to Figure.savefig; dpi is fixed at 160.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, **savefig_options)
    plt.close(fig)


def draw_history(data, path, *, title="Native history"):
    """Render native criticality, burnup and channel-power histories.

    Parameters
    ----------
    data : mapping
        Compact plotting record with sampled times, keff and standard errors, burnup,
        physical channel identities and power summaries.
    path : str or pathlib.Path
        Destination figure file; missing parent directories are created.
    title : str, optional
        Figure heading.
    """
    cycles = data.get("cycles", len(data["cycle_boundaries_days"]))
    days, k = np.array(data["time_days"]), np.array(data["keff"])
    count = len(data["channel_ids"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    axes[0, 0].errorbar(days, k[:, 0], yerr=k[:, 1], fmt=".-", capsize=2)
    limit = data["nominal_keff_limit"]
    axes[0, 0].axhline(limit, color="0.35", ls="--", label=f"Nominal acceptance {limit:.3f}")
    partial = data.get("status", "complete") != "complete"
    sampled = f"{len(k)}/{data['expected_states']} completed states" if partial else f"all {len(k)} states"
    axes[0, 0].set(xlabel="Time (days)", ylabel=r"$k_\mathrm{eff}$", title=f"OpenMC, {sampled}; ±1 SE")
    axes[0, 0].legend(fontsize=8)
    for cycle, boundary in zip(data["burnup"], data["cycle_boundaries_days"]):
        axes[0, 1].plot(cycle["time_days"], cycle["peak_burnup_MWd_tHM"], ".-",
                        color=f"C{cycle['cycle']-1}", label=f"Cycle {cycle['cycle']}")
        axes[0, 0].axvline(boundary, color="0.7", lw=.7)
    if data.get("peak_burnup_limit_MWd_tHM") is not None:
        axes[0, 1].axhline(data["peak_burnup_limit_MWd_tHM"], color="0.35", ls="--")
    bu_title = "Assembly burnup; closed checkpoints" if partial else "Accumulated assembly burnup"
    axes[0, 1].set(xlabel="Time (days)", ylabel="Peak assembly BU (MWd/tHM)", title=bu_title)
    if data["burnup"]:
        axes[0, 1].legend(fontsize=8)
    profile = data.get("temporal_power", data.get(f"cycle{cycles}_temporal_power", data.get("displayed_temporal_power")))
    if profile is not None:
        cycle = data.get("displayed_cycle", cycles)
        qualifier = " (partial)" if data.get("displayed_cycle_partial", False) else ""
        minimax = data.get("power_metric") == "maximum_channel_power_swing_MW"
        power = np.asarray(profile["channel_power_W"])
        swing = np.ptp(power, axis=0)/1e6
        limiting = int(swing.argmax())
        relative = power/1e6 if minimax else 100 * (power / profile["mean_power_W"] - 1)
        for j, name in enumerate(data["channel_names"]):
            axes[1, 0].plot(profile["time_days"], relative[:, j], ".-", label=name,
                            lw=2.2 if minimax and j == limiting else 1.)
        if not minimax:
            axes[1, 0].axhline(0, color="0.5", lw=.7)
        axes[1, 0].set(xlabel=f"Time within cycle {cycle} (days)",
                        ylabel="Total channel power (MW)" if minimax else "Deviation from own mean (%)",
                        title=f"Total radial-channel power; cycle {cycle}{qualifier}")
        axes[1, 0].legend(fontsize=7, ncol=3)
        values = swing if minimax else 100*np.sqrt(profile["variance_by_channel"])
        colours = ["#b75f21" if minimax and j == limiting else "#31738b" for j in range(count)]
        axes[1, 1].bar(range(count), values, color=colours)
        title = (f"Cycle {cycle}{qualifier}; largest swing {swing[limiting]:.3f} MW ({data['channel_names'][limiting]})"
                 if minimax else f"Cycle {cycle}{qualifier}; core RMS {100*np.sqrt(profile['variance']):.2f}%")
        axes[1, 1].set(xticks=range(count), xticklabels=data["channel_names"],
                        ylabel="Maximum − minimum power (MW)" if minimax else "Temporal RMS (%)", title=title)
        if cycle == cycles and "maximum_channel_temporal_rms" in data:
            axes[1, 1].axhline(100*data["maximum_channel_temporal_rms"], color="0.35", ls=":",
                               label="Every-channel limit")
            axes[1, 1].legend(fontsize=8)
    elif partial:
        for ax in axes[1]:
            ax.text(.5, .5, "Power history pending another completed state", ha="center", va="center",
                    transform=ax.transAxes, wrap=True)
            ax.set_axis_off()
    else:
        axes[1, 0].bar(range(count), np.asarray(data["channel_mean_power_W"]) / 1e6, color="#31738b")
        physical_channels = sum(data["channel_physical_multiplicity"])
        axes[1, 0].axhline(data["full_core_power_W"] / physical_channels / 1e6, ls="--", color="0.35")
        axes[1, 0].set(xticks=range(count), xticklabels=data["channel_names"],
                        ylabel="Power per full radial channel (MW)", title="Final-cycle mean channel power")
        field = np.array(data["axial_mean_power_W"]) / 1000
        artist = axes[1, 1].imshow(field, aspect="auto", cmap="viridis")
        axes[1, 1].set(xticks=range(count), xticklabels=data["channel_names"],
                        yticks=range(len(field)), yticklabels=[f"Level {i}" for i in range(len(field))],
                        title="Final-cycle axial power diagnostic")
        fig.colorbar(artist, ax=axes[1, 1], label="Mean assembly power (kW)")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    suffix = f"; {data['status'].replace('_', ' ')}" if partial else ""
    heading = f"{title}: {data['label']}; {cycles} cycles; {data['cycle_days']:.2f} days/cycle{suffix}"
    if partial and data.get("displayed_cycle", 0) < cycles:
        heading += "\nFinal-cycle power results pending"
    fig.suptitle(heading)
    finish(fig, path)


def plot_fit(history, mapping, fit, path, *, title="Current native case", channel_names=None):
    """Compare connected surrogate fields with saved native observations.

    Parameters
    ----------
    history : mapping
        Saved native cycle history containing observations and beginning/midpoint/end state
        keys.
    mapping : FuelMap
        Fuel-region to physical-channel/axial mapping, including symmetry fractions.
    fit : mapping
        Fit record containing selected coupled predictions and field-quality status.
    path : str or pathlib.Path
        Destination figure file; missing parent directories are created.
    title : str, optional
        Figure heading.
    channel_names : mapping or None, optional
        Optional physical channel identifier to display-name mapping.
    """
    keys, days = sample_order(history)
    prediction = fit["selected"]["coupled"]["predictions"]
    names = dict(zip(mapping.channel_ids, mapping.channel_ids)) if channel_names is None else channel_names
    order = [mapping.indices[f"{c}/axial/{z}"] for c in mapping.channel_ids for z in range(mapping.count)]
    labels = [f"{names[c]} {z}" for c in mapping.channel_ids for z in range(mapping.count)]
    groups = np.asarray(history["observations"][keys[0]]["group_flux_cm2_s"]).shape[-1]
    quantities = [("power_W", None, "Assembly power (kW)", 1000.)] + [
        ("group_flux_cm2_s", group, f"Group {group} flux (10¹³ n/cm²/s)", 1e13)
        for group in range(groups)]
    fig, panels = plt.subplots(len(quantities), 2, figsize=(12, 4 * len(quantities)), layout="constrained", squeeze=False)
    for row, (quantity, group, quantity_title, scale) in enumerate(quantities):
        fields = []
        for data in (history["observations"], prediction):
            values = np.array([data[key][quantity] for key in keys])
            field = (values / mapping.fractions.numpy()[None, :]) if group is None else values[:, :, group]
            fields.append(field.T[order] / scale)
        low, high = min(f.min() for f in fields), max(f.max() for f in fields)
        for col, field in enumerate(fields):
            ax = panels[row, col]
            artist = ax.imshow(field, aspect="auto", cmap="viridis", vmin=low, vmax=high)
            ticks = list(range(0, len(keys), 4))
            ax.set(yticks=range(len(labels)), yticklabels=labels, xticks=ticks,
                   xticklabels=[f"{days[i]:.0f}" for i in ticks], xlabel="Sampled state (time in days)",
                   title=("OpenMC: " if col == 0 else "Connected surrogate: ") + quantity_title)
            ax.tick_params(labelsize=7)
            for line in np.arange(1, len(mapping.channel_ids)) * mapping.count - .5:
                ax.axhline(line, color="white", lw=.6)
            for line in [keys.index(cycle["bol"]) - .5 for cycle in history["cycles"][1:]]:
                ax.axvline(line, color="white", lw=.6)
        fig.colorbar(artist, ax=panels[row].tolist(), shrink=.8)
    quality = "Field checks passed" if fit["field_quality_passed"] else "Exploratory fit; strict field criteria unmet"
    fig.suptitle(f"{title}\n"
                 f"{quality}; {len(fit['selected']['parameter_names'])} shared coefficients\n"
                 "All times use physical live-inventory cross sections and connected Bateman depletion")
    finish(fig, path)


def history_data(history, mapping, *, label, nominal_keff_limit, peak_burnup_limit=None):
    """Build figure data from an already completed native history.

    Parameters
    ----------
    history : mapping
        Saved native cycle history containing observations and beginning/midpoint/end state
        keys.
    mapping : FuelMap
        Fuel-region to physical-channel/axial mapping, including symmetry fractions.
    label : str
        Display label identifying this native evaluation.
    nominal_keff_limit : float
        Nominal effective multiplication-factor acceptance threshold.
    peak_burnup_limit : float or None, optional
        Optional peak assembly burnup limit in MWd/tHM.

    Returns
    -------
    dict
        Serializable plotting data with days, watts, MWd/tHM and explicit physical-channel
        multiplicities.
    """
    from coupling.optimization import native_power_profile
    from fuel_cycle.residence import native_temporal_profile
    keys, days = sample_order(history)
    profile = native_power_profile(history, mapping)
    offset, boundaries, burnup = 0., [], []
    for cycle in history["cycles"]:
        burnup.append({"cycle": cycle["cycle"],
                       "time_days": [offset + segment["time_days"] for segment in cycle["segments"]],
                       "peak_burnup_MWd_tHM": [max(segment["burnup_MWd_tHM"]) for segment in cycle["segments"]]})
        offset += cycle["cycle_days"]
        boundaries.append(offset)
    return {"label": label, "status": "complete", "cycles": len(history["cycles"]),
            "time_days": days.tolist(), "keff": [history["observations"][key]["keff"] for key in keys],
            "cycle_boundaries_days": boundaries, "burnup": burnup,
            "cycle_days": history["cycles"][0]["cycle_days"],
            "channel_ids": list(mapping.channel_ids), "channel_names": list(mapping.channel_ids),
            "channel_mean_power_W": profile["channel_mean_power_W"],
            "channel_physical_multiplicity": profile["channel_physical_multiplicity"],
            "full_core_power_W": profile["full_core_power_W"],
            "temporal_power": native_temporal_profile(history, mapping),
            "peak_burnup_limit_MWd_tHM": peak_burnup_limit,
            "nominal_keff_limit": nominal_keff_limit}
