"""Render WR-1 records through shared plotting interfaces; source records are read-only."""
import numpy as np

from runtime.paths import RUNS
from optimization.design import set_parameters
from plotting.case import render_case
from examples.wr_1.model.case import study, new_run

STEPS = (1, 2, 3, 4, 5, 6)
USE_RETAINED_RESULTS = True
OUTPUT = RUNS / "wr_1/figures"


def layout(case, values, native=None):
    """Describe WR-1 cylinders at the middle axial plane for plotting.

    Parameters
    ----------
    case : CycleCase
        WR-1 case adapter supplying the design and native model factories.
    values : dict
        Complete named design parameters in their declared units and array shapes.
    native : dict or None, optional
        Native history with channel mean powers; None colors fuel by fresh enrichment.

    Returns
    -------
    dict
        Ordered cylinder layers and the hexagonal outline in mm; fuel colors are enrichment
        percent or channel power in MW.
    """
    design = case.design()
    set_parameters(design, values)
    mapped = design()
    xy = mapped["channel_xy_mm"].detach().numpy()
    radii = [row.detach().numpy() for row in mapped["channel_interfaces_mm"]]
    layers = []
    if "sleeve_outer_radius_mm" in mapped:
        layers.append({"xy_mm": xy, "outer_radius_mm": mapped["sleeve_outer_radius_mm"][:, design.axial_count // 2].detach().numpy(),
                       "color": "#666c6b"})
    if "graphite_xy_mm" in mapped:
        layers.append({"xy_mm": mapped["graphite_xy_mm"].detach().numpy(),
                       "outer_radius_mm": mapped["graphite_radius_mm"].detach().numpy(), "color": "#666c6b"})
    for shell in reversed(range(max(len(row) for row in radii))):
        selected = [index for index, row in enumerate(radii) if len(row) > shell]
        layers.append({"xy_mm": xy[selected], "outer_radius_mm": [radii[index][shell] for index in selected],
                       "color": ("#dce5e7", "#929da2", "#e9e6dc", "#a2adb2", "#f2f3f1")[shell % 5]})
    active = design.fuelled.numpy()
    colors = 100 * design.enrichment.detach().numpy()[design.ring.numpy()[active]]
    transport = case.make_case(design_values=values)
    if native is not None:
        power = dict(zip(native["power_profile"]["channel_ids"], native["power_profile"]["channel_mean_power_W"]))
        colors = []
        for name in np.asarray(design.channel_ids)[active]:
            index = transport["physical_assembly_to_region"][f"{name}/axial/0"]
            channel = transport["regions"][index]["label"].rpartition("/axial/")[0]
            colors.append(power[channel] / 1e6)
        colors = np.asarray(colors)
    layers.append({"xy_mm": xy[active], "outer_radius_mm": [radii[index][0] for index in np.flatnonzero(active)], "values": colors})
    apothem = transport["geometry"]["apothem_cm"] * 10
    angle = np.arange(6) * np.pi / 3 + np.pi / 6
    outline = apothem / np.cos(np.pi / 6) * np.column_stack((np.cos(angle), np.sin(angle)))
    return {"layers": layers, "outline_mm": outline}



def main():
    """Render the selected WR-1 scenarios and print the figure output directory."""
    for step in STEPS:
        case = study(step) if USE_RETAINED_RESULTS else new_run(step)
        render_case(case, OUTPUT / f"optimization_step_{step}", title=f"WR-1 step {step}",
                    layout=layout, composition_label="Fresh U-235 enrichment (%)")
    print(OUTPUT)


if __name__ == "__main__":
    main()
