"""One zoned pin lattice shared by CAD, OpenMC and diffusion."""

import torch

from geometry import CadGeometry, MaterialVolume, box, cylinder, union
from geometry.channels import rectangular_centers


def pin_centers(settings):
    """Locate the square lattice in its common physical coordinate system.

    Parameters
    ----------
    settings : dict
        Geometry inputs with pin counts and pitches in cm.

    Returns
    -------
    torch.Tensor
        Pin-by-coordinate centers in cm; x varies fastest.
    """
    return rectangular_centers(settings["shape"], settings["pitch_cm"])


def build_geometry(settings):
    """Construct fuel segments and the water filling the reflected domain.

    Parameters
    ----------
    settings : dict
        Physical geometry from model.toml. All dimensions use cm.

    Returns
    -------
    geometry.CadGeometry
        Fuel regions ordered by pin then axial segment, followed by water.
        Axial material interfaces belong to the physical CAD construction.

    Notes
    -----
    Bare UO2 fuel cylinders are an explicit illustrative simplification: no
    cladding, gaps, guide tubes or structural supports are represented.
    """
    half = (torch.tensor(settings["shape"], dtype=torch.float64)
            * torch.as_tensor(settings["pitch_cm"], dtype=torch.float64) / 2)
    rx, ry, rz = settings["reflector_cm"]
    height = settings["fuel_height_cm"]
    domain = box([-half[0] - rx, -half[1] - ry, -rz],
                 [half[0] + rx, half[1] + ry, height + rz])
    edges = torch.linspace(0., height, settings["axial_segments"] + 1, dtype=torch.float64)
    pins = tuple(cylinder(xy, settings["fuel_radius_cm"], [0., height])
                 for xy in pin_centers(settings))
    volumes = tuple(MaterialVolume(f"fuel_{i:02d}_{j:02d}", "fuel", pin.slice(edges[j:j+2]))
                    for i, pin in enumerate(pins) for j in range(len(edges) - 1))
    volumes += (MaterialVolume("water", "water", domain - union(pins)),)
    names = ("x_min", "x_max", "y_min", "y_max", "bottom", "top")
    boundaries = {name: {"type": kind, "slot": i}
                  for i, (name, kind) in enumerate(zip(names, settings["boundaries"]))}
    return CadGeometry(domain, volumes, boundaries, "enrichment_shaping")
