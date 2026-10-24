"""Rotations and labelled orbit checks for sixfold spatial symmetry."""
import math

import numpy as np

from diffusion_surrogate.geometry import GeometryInputError


class SectorInputError(GeometryInputError):
    """Declared sector geometry, quadrature or symmetry is infeasible."""


def rotate(vector, sixths):
    """Rotate x/y coordinates by an integer number of sixty-degree turns.

    Parameters
    ----------
    vector : array_like
        Coordinate vector or array with x and y in its first two final-axis entries.
    sixths : int
        Counterclockwise rotation count in units of sixty degrees.

    Returns
    -------
    numpy.ndarray
        Rotated floating-point coordinates, leaving any additional components unchanged.
    """
    angle = int(sixths) * math.pi / 3
    c, s = math.cos(angle), math.sin(angle)
    value = np.array(vector, dtype=float, copy=True)
    value[..., 0] = c * np.asarray(vector)[..., 0] - s * np.asarray(vector)[..., 1]
    value[..., 1] = s * np.asarray(vector)[..., 0] + c * np.asarray(vector)[..., 1]
    return value


def audit_c6_orbits(xy, labels, *, tolerance_cm):
    """Verify coordinate and material-state equivalence under sixfold rotation.

    Parameters
    ----------
    xy : array_like
        Full-layout x/y site coordinates in cm, shaped (n_sites, 2).
    labels : sequence
        Comparable per-site identities encoding all relevant composition, geometry and
        schedule state.
    tolerance_cm : float
        Positive distance tolerance in cm for matching rotational partners.

    Returns
    -------
    tuple of tuple of int
        Disjoint site-index orbits, each containing the central site or six partners.

    Raises
    ------
    SectorInputError
        Coordinates, partner uniqueness or state labels violate the declared C6 symmetry.

    Notes
    -----
    Labels must encode every state that matters (axial stack, composition, control and
    refuelling phase). This geometric check cannot infer those data. Axis sites have
    multiplicity one; other complete orbits have six members.
    """
    xy = np.asarray(xy, dtype=float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(labels)
            or not np.isfinite(xy).all() or tolerance_cm <= 0
            or not math.isfinite(tolerance_cm)):
        raise SectorInputError("Supply finite coordinates, one label per site and positive tolerance")
    unseen, orbits = set(range(len(xy))), []
    while unseen:
        first = min(unseen)
        orbit = set()
        for k in range(6):
            matches = np.flatnonzero(np.linalg.norm(xy - rotate(xy[first], k), axis=1) <= tolerance_cm)
            if len(matches) != 1:
                raise SectorInputError("Layout has a missing or ambiguous C6 partner")
            index = int(matches[0])
            if labels[index] != labels[first]:
                raise SectorInputError("Material, inventory or schedule labels violate C6 symmetry")
            orbit.add(index)
        if len(orbit) not in (1, 6) or (len(orbit) == 1 and np.linalg.norm(xy[first]) > tolerance_cm):
            raise SectorInputError("Invalid rotational orbit")
        if not orbit <= unseen:
            raise SectorInputError("Overlapping rotational orbits")
        unseen -= orbit
        orbits.append(tuple(sorted(orbit)))
    return tuple(orbits)
