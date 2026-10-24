"""Shared geometry descriptions and physical-to-mesh mappings."""
from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


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
    points = np.asarray(vertices, dtype=float)-np.asarray(center, dtype=float)
    area = 0.
    for first, second in zip(points, np.roll(points, -1, axis=0)):
        delta = second-first
        qa, qb, qc = float(delta@delta), 2*float(first@delta), float(first@first)-radius*radius
        cuts = [0., 1.]
        discriminant = qb*qb-4*qa*qc
        if discriminant > 0:
            root = math.sqrt(discriminant)
            cuts += [t for t in ((-qb-root)/(2*qa), (-qb+root)/(2*qa)) if 0 < t < 1]
        cuts.sort()
        for lower, upper in zip(cuts[:-1], cuts[1:]):
            left, right = first+lower*delta, first+upper*delta
            cross = float(left[0]*right[1]-left[1]*right[0])
            middle = first+(lower+upper)/2*delta
            area += (cross/2 if middle@middle < radius*radius
                     else radius*radius*math.atan2(cross, float(left@right))/2)
    return abs(area)
