"""One parametric build123d lattice definition for both numerical adapters."""

import torch

from geometry import CadGeometry, MaterialVolume, box, cylinder, union


def pin_centers(settings):
    """Place the square lattice symmetrically around the origin.

    Parameters
    ----------
    settings : dict
        Geometry settings with pin counts and x/y pitches in cm.

    Returns
    -------
    torch.Tensor
        Pin-by-coordinate centers in cm, with x varying fastest.
    """
    nx, ny = settings["shape"]
    px, py = settings["pitch_cm"]
    return torch.tensor([[(i - (nx - 1) / 2) * px, (j - (ny - 1) / 2) * py]
                         for j in range(ny) for i in range(nx)], dtype=torch.float64)


def build_geometry(settings, radius_cm):
    """Build individually named fuel cylinders and their surrounding coolant.

    Parameters
    ----------
    settings : dict
        Geometry dimensions in cm and the six outer boundary conditions.
    radius_cm : torch.Tensor
        Live common fuel radius in cm.

    Returns
    -------
    geometry.CadGeometry
        Shared parametric solids, with fuels first and coolant last. The same
        construction builds CAD for OpenMC and analytical material fractions.
    """
    half = (torch.tensor(settings["shape"], dtype=torch.float64)
            * torch.tensor(settings["pitch_cm"], dtype=torch.float64) / 2)
    height = settings["height_cm"]
    domain = box([-half[0], -half[1], 0.], [half[0], half[1], height])
    pins = tuple(cylinder(xy, radius_cm, [0., height]) for xy in pin_centers(settings))
    volumes = tuple(MaterialVolume(f"fuel_{i:02d}", "fuel", pin) for i, pin in enumerate(pins))
    volumes += (MaterialVolume("coolant", "coolant", domain - union(pins)),)
    faces = ("x_min", "x_max", "y_min", "y_max", "bottom", "top")
    boundaries = {face: {"type": boundary, "slot": index}
                  for index, (face, boundary) in enumerate(zip(faces, settings["boundaries"]))}
    return CadGeometry(domain, volumes, boundaries, "small_lattice")
