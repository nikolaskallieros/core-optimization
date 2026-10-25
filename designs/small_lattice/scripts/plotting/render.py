"""Compare the initial and selected lattice geometry and criticality histories."""

from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from ...build_geometry import pin_centers

FIGURE = Path(__file__).resolve().parents[2] / "results/lattice_optimization/optimization_step_1/figures/workflow.png"


def render(record):
    """Render geometry and histories from a completed, explicit run record.

    Parameters
    ----------
    record : dict
        Passed verification record, including its native histories and input path.

    Returns
    -------
    pathlib.Path
        Retained figure path under the design's results directory.
    """
    directory = Path(record["directory"])
    settings = tomllib.loads((directory / "model.toml").read_text())
    geometry = settings["geometry"]
    initial = record["references"][0]
    final = next(reference for reference in record["references"]
                 if reference["directory"] == record["selected_native_directory"])
    half = np.asarray(geometry["shape"]) * np.asarray(geometry["pitch_cm"]) / 2
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.45]},
                             layout="constrained")
    for ax, reference, name in zip(axes[:2], (initial, final), ("Initial lattice", "Optimized lattice")):
        ax.set_facecolor("#edf3f6")
        for x, y in pin_centers(geometry).numpy():
            ax.add_patch(Circle((x, y), reference["radius_cm"], facecolor="#377b9d",
                                edgecolor="#174258", linewidth=.8))
        for axis, count in zip(("x", "y"), settings["mesh"]["subdivisions"][:2]):
            extent = half[0 if axis == "x" else 1]
            for value in np.linspace(-extent, extent, count + 1):
                (ax.axvline if axis == "x" else ax.axhline)(value, color="#c3ccd1", lw=.35, alpha=.6)
        ax.set(xlim=(-half[0], half[0]), ylim=(-half[1], half[1]), aspect="equal",
               xlabel="x (cm)", ylabel="y (cm)",
               title=f"{name}\nr = {reference['radius_cm']:.4f} cm")
    ax = axes[2]
    for reference, label, color in ((initial, "Initial: OpenMC", "#777777"),
                                    (final, "Optimized: OpenMC", "#236c93")):
        history = reference["history"]
        ax.errorbar(history["time_days"], history["keff_history"],
                    yerr=history["standard_error_history"], marker="o", markersize=4,
                    color=color, label=label, capsize=3, lw=1.4)
    surrogate = record["checks"]["surrogate_history"]
    ax.plot(surrogate["time_days"], surrogate["keff_history"], "--", color="#bf7b22",
            label="Optimized: diffusion", lw=1.5)
    ax.axhline(settings["optimization"]["target_keff"], color="#333333", linestyle=":", lw=1.2)
    duration = sum(settings["history"]["durations_days"])
    ax.set(xlabel="Time (days)", ylabel=r"$k_{\mathrm{eff}}$", title=f"{duration:g}-day depletion history")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=.2)
    fig.suptitle("Small lattice · manufactured one-group data", fontsize=13)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180)
    fig.savefig(directory / "workflow.png", dpi=180)
    plt.close(fig)
    return FIGURE
