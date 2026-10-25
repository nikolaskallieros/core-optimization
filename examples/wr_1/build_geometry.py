"""Build the WR-1 CAD assembly from the resolved TOML-driven design."""

from geometry.channels import Cylinder, channel_geometry, hexagonal_domain

STEP = 1


def parameterized_geometry(design, case):
    """Construct the shared WR-1 CAD model from the live TOML-driven design.

    Parameters
    ----------
    design : Design
        Live geometry and optional graphite parameters.
    case : dict
        Material identities, axial intervals and physical boundary conditions.

    Returns
    -------
    geometry.CadGeometry
        One build123d construction used by both OpenMC and diffusion adapters.
    """
    mapped = design()
    cylinders = {}
    z = design.z_edges_mm.detach().numpy() / 10
    for i, site in enumerate(design.channel_ids):
        radii = tuple(mapped["channel_interfaces_mm"][i] / 10)
        if "sleeve_outer_radius_mm" in mapped and bool(design.fuelled[i]):
            by_level = mapped["sleeve_outer_radius_mm"][i] / 10
            radii += ({(float(z[-2-level]), float(z[-1-level])): by_level[level]
                       for level in range(design.axial_count)},)
        cylinders[site] = Cylinder(mapped["channel_xy_mm"][i] / 10, radii)
    if "graphite_xy_mm" in mapped:
        cylinders.update({site: Cylinder(xy / 10, radius.reshape(1) / 10)
                          for site, xy, radius in zip(design.graphite_ids, mapped["graphite_xy_mm"], mapped["graphite_radius_mm"])})
    domain, boundaries = hexagonal_domain(case["geometry"])
    return channel_geometry(domain, case["regions"], cylinders, boundaries, name="WR-1")

def build_geometry(design, case):
    """Construct named build123d material solids for the selected WR-1 design.

    Parameters
    ----------
    design : examples.wr_1.model.design.Design
        Physical dimensions and design variables resolved from TOML.
    case : dict
        Ordered material identities and boundary choices.

    Returns
    -------
    geometry.CadGeometry
        Actual build123d solids in mm with materials and boundary-face selections.
        Meshing and solver conversion belong to the shared adapters in src.
    """
    return parameterized_geometry(design, case).as_solids()


if __name__ == "__main__":
    from examples.wr_1.model.case import study

    scenario = study(STEP)
    build_geometry(scenario.design(), scenario.make_case()).write_step(scenario.work / "geometry.step")
