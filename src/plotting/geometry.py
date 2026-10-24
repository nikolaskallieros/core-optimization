"""Geometry figures from caller-defined circle layers and parameter labels."""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from plotting.history import finish


def plot_layouts(layouts, path, *, title, color_label="", labels=None):
    """Draw caller-defined cylindrical geometry layers.

    Parameters
    ----------
    layouts : sequence of mapping
        Layouts containing outline_mm and circle layers with xy_mm, outer_radius_mm,
        optional inner_radius_mm, and values or a fixed color.
    path : str or pathlib.Path
        Destination figure file; missing parent directories are created.
    title : str
        Figure heading.
    color_label : str, optional
        Shared colorbar label, including units where applicable.
    labels : sequence of str or None, optional
        Optional panel heading for each supplied layout.
    """
    values = [np.asarray(layer["values"]) for layout in layouts for layer in layout["layers"]
              if "values" in layer]
    norm = Normalize(min(value.min() for value in values), max(value.max() for value in values)) if values else None
    cmap = plt.get_cmap("viridis")
    fig, panels = plt.subplots(1, len(layouts), figsize=(6 * len(layouts), 6), layout="constrained", squeeze=False)
    for index, (axis, layout) in enumerate(zip(panels.flat, layouts)):
        for layer in layout["layers"]:
            for i, (xy, outer) in enumerate(zip(layer["xy_mm"], layer["outer_radius_mm"])):
                color = cmap(norm(layer["values"][i])) if "values" in layer else layer.get("color", "#b6bfc2")
                axis.add_patch(Circle(xy, outer, facecolor=color, edgecolor="#46545d", lw=.35))
                inner = layer.get("inner_radius_mm")
                if inner is not None and inner[i] > 0:
                    axis.add_patch(Circle(xy, inner[i], facecolor="white", edgecolor="#46545d", lw=.3))
        outline = np.asarray(layout["outline_mm"])
        axis.plot(*np.vstack((outline, outline[0])).T, color=".3", lw=.7)
        axis.set(aspect="equal", xlabel="x (mm)", ylabel="y (mm)",
                 title=labels[index] if labels else "")
        axis.autoscale_view()
    if norm is not None:
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(panels.flat),
                     label=color_label, shrink=.7)
    fig.suptitle(title)
    finish(fig, path)


def plot_parameter_changes(initial, final, path, *, title="Design changes", labels=None):
    """Compare initial and latest design components with paired bars.

    Parameters
    ----------
    initial : mapping
        Initial parameter arrays in their original design units.
    final : mapping
        Latest parameter arrays in the same units.
    path : str or pathlib.Path
        Destination figure file; missing parent directories are created.
    title : str, optional
        Figure heading.
    labels : mapping or None, optional
        Optional parameter-to-component labels; absent labels use component indices.
    """
    labels = {} if labels is None else labels
    rows = math.ceil(len(final) / 2)
    fig, panels = plt.subplots(rows, 2, figsize=(11, 3 * rows), layout="constrained", squeeze=False)
    for axis, (name, values) in zip(panels.flat, final.items()):
        values = np.asarray(values).reshape(-1)
        x = np.arange(len(values))
        if name in initial:
            axis.bar(x - .18, initial[name], width=.35, color="#a4aaae", label="Initial")
        axis.bar(x + .18, values, width=.35, color="#31738b", label="Latest")
        axis.set(title=name.replace("_", " "), xticks=x,
                 xticklabels=labels.get(name, [str(i) for i in x]))
        axis.tick_params(axis="x", rotation=60 if len(x) > 8 else 0, labelsize=7)
        axis.legend(fontsize=7)
    for axis in list(panels.flat)[len(final):]:
        axis.set_visible(False)
    fig.suptitle(title)
    finish(fig, path)
