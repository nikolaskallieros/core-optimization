"""Compare native pin/axial powers and the physical enrichment field."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DESIGN_ROOT = Path(__file__).resolve().parents[2]
FIGURES = DESIGN_ROOT / "results/power_shaping/optimization_step_1/figures"


def render(record):
    """Plot retained native powers without rerunning transport or diffusion.

    Parameters
    ----------
    record : dict
        Completed preparation or optimization summary with effective inputs,
        native evaluations and the selected native index.

    Returns
    -------
    list of str
        Figure paths written under this design's versioned figure directory.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    config = record["config"]
    nx, ny = config["geometry"]["shape"]
    zones = config["geometry"]["axial_segments"]
    height = config["geometry"]["fuel_height_cm"]
    centers = (np.arange(zones) + .5) * height / zones
    baseline = record["native"][0]
    selected = record["native"][record["selected_index"]]
    comparison = selected["index"] != baseline["index"]
    points = [baseline, selected] if comparison else [baseline]
    initial = config["materials"]["initial_enrichment_wt_percent"]
    labels = ["Initial", "Selected"] if comparison else [f"Uniform {initial:g} wt% reference"]
    fields = [np.asarray(point["relative_power_density"]).reshape(ny, nx, zones) for point in points]
    figure, axes = plt.subplots(len(points), 2, figsize=(10, 4.1 * len(points)), squeeze=False,
                                constrained_layout=True)
    maximum = max(field.max() for field in fields)
    for row, (point, field, label) in enumerate(zip(points, fields, labels)):
        axis = axes[row, 0]
        image = axis.imshow(field[:, :, zones // 2], origin="lower", cmap="viridis", vmin=0., vmax=maximum)
        figure.colorbar(image, ax=axis, label="Local / average fuel power density", shrink=.85)
        axis.set(xlabel="Pin x index", ylabel="Pin y index",
                 title=f"{label}: z = {centers[zones // 2]:g} cm")
        axis = axes[row, 1]
        axis.plot(centers, field.mean((0, 1)), "o-", color="#2a6f97", label="Mean over pins")
        axis.plot(centers, field.max((0, 1)), "s-", color="#c1662b", label="Hottest pin at each height")
        axis.axhline(1., color=".5", linestyle="--", linewidth=1)
        axis.set(xlabel="Axial position (cm)", ylabel="Local / average fuel power density",
                 ylim=(0., maximum * 1.12),
                 title=(f"Peak/average = {point['peaking_factor']:.3f}; "
                        f"k = {point['keff']:.5f} ± {point['standard_error']:.5f}"))
        axis.legend(frameon=False, fontsize=9)
        axis.grid(alpha=.2)
    figure.suptitle("Pin and axial power distribution · continuous-energy OpenMC", fontsize=14)
    path = FIGURES / ("power_profiles.png" if comparison else "reference_power.png")
    figure.savefig(path, dpi=170)
    plt.close(figure)
    paths = [str(path)]
    if comparison:
        enrichment = np.asarray(selected["enrichment_wt_percent"]).reshape(ny, nx, zones)
        figure, axes = plt.subplots(2, int(np.ceil(zones / 2)), figsize=(13, 5.2),
                                    squeeze=False, constrained_layout=True)
        low, high = config["materials"]["enrichment_bounds_wt_percent"]
        for index, axis in enumerate(axes.flat):
            if index >= zones:
                axis.set_visible(False)
                continue
            image = axis.imshow(enrichment[:, :, index], origin="lower", cmap="viridis", vmin=low, vmax=high)
            axis.set_title(f"z = {centers[index]:g} cm")
            axis.set_xticks([0, nx // 2, nx - 1])
            axis.set_yticks([0, ny // 2, ny - 1])
        figure.colorbar(image, ax=axes.ravel().tolist(), label="U-235 enrichment (wt% within uranium)", shrink=.85)
        initial = config["materials"]["initial_enrichment_wt_percent"]
        figure.suptitle(f"Selected enrichment · {enrichment.size} parameters · initial {initial:g} wt% everywhere", fontsize=14)
        path = FIGURES / "enrichment.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(str(path))
    if record.get("proposals"):
        paths.append(plot_progress(record))
    return paths


def plot_progress(record):
    """Compare surrogate iterations with independently verified native progress.

    Parameters
    ----------
    record : dict
        Native evaluations, proposal decisions and retained optimizer traces.

    Returns
    -------
    str
        Saved progress figure. Missing iterations from a resumed checkpoint are
        shown as endpoints, without inventing an intermediate history.
    """
    points = record["native"]
    proposals = record["proposals"]
    config = record["config"]["optimization"]
    steps = np.arange(len(points))
    peaks = np.array([point["peaking_factor"] for point in points])
    accepted = [0] + [i + 1 for i, proposal in enumerate(proposals) if proposal["accepted"]]
    rejected = [i + 1 for i, proposal in enumerate(proposals) if not proposal["accepted"]]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axis = axes[0]
    incumbent = np.array([peaks[max(index for index in accepted if index <= step)] for step in steps])
    axis.step(steps, incumbent, where="post", color="#2a6f97", linewidth=2, label="Accepted design")
    axis.scatter(accepted, peaks[accepted], color="#2a6f97", s=35, zorder=3)
    axis.scatter(rejected, peaks[rejected], marker="x", s=85, linewidth=2,
                 color="#bb4b36", label="Rejected trial", zorder=3)
    for step in accepted:
        peak = peaks[step]
        axis.annotate(f"{peak:.3f}", (step, peak), xytext=(0, 10),
                      textcoords="offset points", ha="center", fontsize=10)
    axis.set(xlabel="OpenMC verification", ylabel="Peak / average fuel power density",
             title="OpenMC power flattening", xticks=steps)
    axis.margins(y=.18, x=.12)
    axis.legend(frameon=False, fontsize=9)

    axis = axes[1]
    target = record["target_keff"]
    reference_error = points[0]["standard_error"] if config["keff_reference"] == "baseline" else 0.
    delta = 1e5 * (np.array([point["keff"] for point in points]) - target)
    errors = 1e5 * config["native_sigma_multiplier"] * np.hypot(
        [point["standard_error"] for point in points], reference_error)
    if config["keff_reference"] == "baseline":
        errors[0] = 0.
    limit = 1e5 * config["keff_absolute_tolerance"]
    axis.axhspan(-limit, limit, color="#2a6f97", alpha=.08)
    axis.axhline(0., color=".5", linewidth=1)
    for value in (-limit, limit):
        axis.axhline(value, color=".5", linewidth=1, linestyle="--")
    axis.errorbar(accepted, delta[accepted], yerr=errors[accepted], fmt="o",
                   color="#2a6f97", capsize=4)
    axis.errorbar(rejected, delta[rejected], yerr=errors[rejected], fmt="x",
                   color="#bb4b36", capsize=4, markersize=8)
    axis.set(xlabel="OpenMC verification", ylabel="Change from reference k (pcm)",
             title=f"Fixed-k acceptance · ±{limit:g} pcm", xticks=steps,
             ylim=(-1.25 * limit, 1.25 * limit))
    axis.text(.5, .97, "Error bars: two combined standard errors",
              transform=axis.transAxes, ha="center", va="top", fontsize=8)
    mass = np.array([point["u235_mass_kg"] for point in points])
    mass_error = np.max(np.abs(mass / mass[0] - 1))
    mass_tolerance = record["config"]["verification"]["inventory_relative_tolerance"]
    axis.text(.5, .02, f"Max relative U-235 error: {mass_error:.1e} (limit {mass_tolerance:.0e})",
              transform=axis.transAxes, ha="center", va="bottom", fontsize=8)

    axis = axes[2]
    colors = plt.get_cmap("tab10")(np.arange(len(proposals)) % 10)
    for index, proposal in enumerate(proposals):
        search = proposal["optimization"]
        previous = search.get("previous_iterations", 0)
        if previous:
            axis.scatter([0, previous], [search["objective_before"], search["objective_after"]],
                         color=colors[index % len(colors)], s=35,
                         label=f"Round {index + 1}: saved endpoints")
            axis.scatter([previous], [search["objective_after"]], marker="*", s=75,
                         color=colors[index % len(colors)], edgecolor="white", linewidth=.5, zorder=4)
        else:
            trace = [entry for entry in search["trace"] if entry.get("phase", "optimization") == "optimization"]
            axis.plot(np.arange(len(trace)), [entry["objective"] for entry in trace],
                      color=colors[index % len(colors)], linewidth=1.8, label=f"Round {index + 1}")
            selected_trace = search.get("selected_trace_index", len(search["trace"]) - 1)
            chosen = search["trace"][selected_trace]
            iteration = sum(entry.get("phase", "optimization") == "optimization"
                            for entry in search["trace"][:selected_trace + 1]) - 1
            axis.scatter([iteration], [chosen["objective"]], marker="*", s=75,
                         color=colors[index % len(colors)], edgecolor="white", linewidth=.5, zorder=4)
    axis.set(xlabel="Diffusion optimizer iteration", ylabel="Smooth peak / average objective",
             title="Calibrated diffusion predictions")
    axis.plot([], [], marker="*", markersize=9, linestyle="none", color=".4", label="Selected feasible iterate")
    axis.legend(frameon=False, fontsize=8, ncol=2 if len(proposals) > 5 else 1)
    for axis in axes:
        axis.grid(alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
    selected = record["selected_index"]
    reduction = 100 * (1 - peaks[selected] / peaks[0])
    parameters = np.asarray(points[0]["enrichment_wt_percent"]).size
    figure.suptitle(f"{parameters} enrichment variables · {reduction:.1f}% lower verified power peak · fixed U-235 inventory",
                   fontsize=14)
    path = FIGURES / "optimization_progress.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return str(path)
