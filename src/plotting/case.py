"""Figures for saved fuel-cycle cases, with caller-defined geometry layers."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from fuel_cycle.residence import FuelMap
from runtime.io import load_state
from plotting.history import draw_history, history_data, plot_fit
from plotting.optimization import (
    native_cases, plot_gradients, plot_search, plot_native_comparison, plot_temporal_comparison,
)
from plotting.geometry import plot_layouts, plot_parameter_changes


def render_case(case, output, *, title="Fuel-cycle optimization", layout=None,
                composition_label="Fresh composition", power_label="Mean channel power (MW)"):
    """Render a case's saved optimization, fit and native figures.

    Parameters
    ----------
    case : CycleCase
        Case supplying settings, saved result paths and native-state paths.
    output : str or pathlib.Path
        Destination directory for the generated figures.
    title : str, optional
        Figure heading.
    layout : callable or None, optional
        Optional callback layout(case, design, native=None) returning generic circle layers
        and an outline.
    composition_label : str, optional
        Colorbar label for the fresh-composition geometry view.
    power_label : str, optional
        Colorbar label for native-power geometry comparisons.

    Returns
    -------
    pathlib.Path
        Output directory containing the written figures.
    """
    output = Path(output)
    options = case.settings("optimization")
    record = json.loads(case.result("optimization").read_text())
    for path in sorted(case.data.glob("native_history_*.json")):
        data = json.loads(path.read_text())
        draw_history(data, output / data["plot"], title=title)
    history = load_state(case.result("history"))
    native = load_state(case.native_states() / f"{history['cycles'][0]['bol']}.json")["case"]
    regions = [region for region in native["regions"] if region["depletable"]]
    mapping = FuelMap(SimpleNamespace(regions=regions, labels=[region["label"] for region in regions],
        multiplicity=np.array([region["physical_multiplicity"] for region in regions]),
        symmetry_multiplier=native["symmetry_multiplier"]))
    if not list(case.data.glob("native_history_*.json")):
        data = history_data(history, mapping, label="Current native case",
                            nominal_keff_limit=options["native_minimum_keff"],
                            peak_burnup_limit=case.settings()["objective"].get("peak_burnup_MWd_tHM"))
        draw_history(data, output / "history.png", title=title)
    fit = json.loads(case.result("fit").read_text())
    plot_fit(history, mapping, fit, output / "fit.png", title=title)
    plot_search(record, options, output / "optimization.png", title=title)
    plot_native_comparison(record, output / "native_comparison.png", title=title)
    plot_temporal_comparison(record, output / "temporal_power.png", title=title)
    steps = [row for batch in record["batches"] for row in batch["steps"]]
    gradient = steps[-1]["gradient"] if steps else json.loads(case.result("gradient_checks").read_text())["gradient"]
    plot_gradients(gradient, output / "gradients.png", title=title + ": full-residence derivatives")
    initial, latest = record["initial_native"], native_cases(record)[-1][1]
    plot_parameter_changes(initial["design"], latest["design"], output / "design_changes.png", title=title)
    if layout is not None:
        plot_layouts([layout(case, row["design"], row) for row in (initial, latest)],
                     output / "design_native_power.png", title=title, color_label=power_label,
                     labels=("Initial native design", "Latest native design"))
        plot_layouts([layout(case, initial["design"])], output / "geometry.png", title=title,
                     color_label=composition_label)
    return output
