"""Shared geometry descriptions and physical-to-mesh mappings."""
from dataclasses import dataclass
from typing import Callable

import torch
from geometry.intersections import circle_triangle_area


@dataclass(frozen=True)
class DiffusionGeometry:
    """One physical region layout discretized on a diffusion mesh.

    Parameters
    ----------
    grid : object
        Solver-compatible mesh carrying cell volumes and connectivity.
    fractions : callable
        Zero-argument callback returning live cell-by-region volume fractions.
    region_labels : tuple of str
        Physical material-region identities in fraction-column order.
    identity : object or None, optional
        Comparable identity of the mesh and mapping, enabling reuse across native states
        with the same live design. None disables shared-geometry caching.
    volume_rtol : float, optional
        Declared relative mesh-volume error allowed by coupled case preparation.
        The default retains exact analytical volume closure. Curved tetrahedral
        meshes need an explicit tolerance justified by mesh refinement; this
        never rescales physical material volumes or densities.

    Notes
    -----
    The case's default mapper discretizes its native physical geometry. An explicitly
    selected alternative mapper may simplify that geometry while preserving region
    identities and represented material volumes. Mesh refinement alone is not a change
    of physical geometry. The mapper owns its coordinate and boundary conventions.
    """
    grid: object
    fractions: Callable
    region_labels: tuple[str, ...]
    identity: object | None = None
    volume_rtol: float = 1e-10


class GeometryInputError(ValueError):
    """Invalid geometry inputs."""


def circle_polygon_area(center, radius, vertices):
    """Integrate the exact intersection of a circle and convex polygon.

    Parameters
    ----------
    center : array_like
        Circle center in the same length unit as the vertices.
    radius : float
        Circle radius in that length unit.
    vertices : array_like
        Ordered vertices of the convex polygon, shaped (n_vertices, 2).

    Returns
    -------
    float
        Nonnegative intersection area in the squared input length unit.

    Notes
    -----
    Circular arcs and straight chords are integrated separately. This geometry calculation
    introduces no transport-volume correction or density scaling.
    """
    return float(circle_triangle_area(torch.as_tensor(vertices, dtype=torch.float64)[None],
                                      torch.as_tensor(center, dtype=torch.float64),
                                      torch.as_tensor(radius, dtype=torch.float64)).abs()[0])
