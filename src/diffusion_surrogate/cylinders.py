"""Cylindrical material regions and live fractions in hexagonal domains."""
from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from diffusion_surrogate.geometry import circle_polygon_area
from diffusion_surrogate.hexagonal import circle_triangle_area
from diffusion_surrogate.solver import tensor


@dataclass(frozen=True)
class HexagonalDomain:
    """A complete regular hexagon or its positive-x, 60-degree sector."""
    apothem_cm: float
    full: bool = False

    @property
    def multiplier(self):
        """Return the physical multiplicity of the represented symmetry domain.

        Returns
        -------
        int
            One for a full hexagon or six for a single sector.
        """
        return 1 if self.full else 6

    @property
    def polygon(self):
        """Return the positive-x triangular symmetry-sector polygon.

        Returns
        -------
        numpy.ndarray
            Three counterclockwise x/y vertices in cm, regardless of the full-domain flag.
        """
        a = self.apothem_cm
        return np.array([[0., 0.], [a, -a / math.sqrt(3)], [a, a / math.sqrt(3)]])

    @property
    def area(self):
        """Compute the transverse area of the represented domain.

        Returns
        -------
        float
            Full-hexagon or single-sector area in cm².
        """
        return (6 if self.full else 1) * self.apothem_cm * (self.apothem_cm / math.sqrt(3))

    def disk_area(self, xy, radius):
        """Compute a disk's represented area under the domain symmetry.

        Parameters
        ----------
        xy : array_like
            Disk center in cm.
        radius : float
            Disk radius in cm.

        Returns
        -------
        float
            Disk-sector intersection area in cm²; full-domain mode assumes the disk is inside
            the enclosing hexagon.
        """
        return math.pi * radius * radius if self.full else circle_polygon_area(xy, radius, self.polygon)

    def bounds(self, z_edges_cm):
        """Return the Cartesian bounding box for the extruded domain.

        Parameters
        ----------
        z_edges_cm : array_like
            Increasing axial interval edges in cm.

        Returns
        -------
        tuple
            Lower and upper three-dimensional coordinate lists in cm.
        """
        a, h = self.apothem_cm, self.apothem_cm / math.sqrt(3)
        return ([-a, -2 * h, float(z_edges_cm[0])] if self.full else [0., -h, float(z_edges_cm[0])],
                [a, 2 * h, float(z_edges_cm[-1])] if self.full else [a, h, float(z_edges_cm[-1])])


def cylinder_clearances(domain, xy_cm, outer_radius_cm, *, obstacles=None):
    """Compute minimum cylinder-pair, wall and obstacle clearances.

    Parameters
    ----------
    domain : HexagonalDomain
        Complete hexagon or one-sixth sector with its declared apothem.
    xy_cm : array_like
        Physical site x/y coordinates in cm, shaped (n_sites, 2).
    outer_radius_cm : array_like
        Outer radius in cm for each site, optionally with a trailing axial-interval axis.
    obstacles : tuple or None, optional
        Optional obstacle centers and outer radii in cm, with compatible axial intervals.

    Returns
    -------
    dict of str to float
        Minimum pair and wall gaps in cm, plus obstacle gap when supplied. Negative gaps
        indicate overlap.
    """
    xy = np.asarray(xy_cm)
    radii = np.asarray(outer_radius_cm)
    if radii.ndim == 1:
        radii = radii[:, None]
    distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    normals = np.array([(math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)) for i in range(6)])
    result = {"pair": float((distances[..., None] - radii[:, None] - radii[None, :]).min()),
              "wall": float((domain.apothem_cm - (xy @ normals.T)[..., None] - radii[:, None]).min())}
    if obstacles is not None:
        other_xy, other_radii = (np.asarray(value) for value in obstacles)
        if other_radii.ndim == 1:
            other_radii = other_radii[:, None]
        gap = np.linalg.norm(xy[:, None] - other_xy[None, :], axis=-1)[..., None]
        result["obstacles"] = float((gap - radii[:, None] - other_radii[None, :]).min())
    return result


def orbit_annulus(domain, site_ids, xy_cm, inner_radius_cm, outer_radius_cm, orbit):
    """Clip one rotational orbit and verify its represented annular area.

    Parameters
    ----------
    domain : HexagonalDomain
        Complete hexagon or one-sixth sector with its declared apothem.
    site_ids : sequence of str
        Stable physical cylinder-site identities.
    xy_cm : array_like
        Physical site x/y coordinates in cm, shaped (n_sites, 2).
    inner_radius_cm : array_like
        Inner annular radius in cm for each physical site.
    outer_radius_cm : array_like
        Outer annular radius in cm for each physical site.
    orbit : sequence of int
        Physical site indices forming one audited symmetry orbit.

    Returns
    -------
    tuple
        Visible annular-piece records and total represented area in cm².

    Raises
    ------
    ArithmeticError
        Visible annular area disagrees with physical orbit multiplicity.
    """
    pieces, area = [], 0.
    for index in orbit:
        inner, outer = float(inner_radius_cm[index]), float(outer_radius_cm[index])
        part = (math.pi * (outer * outer - inner * inner) if domain.full else
                domain.disk_area(xy_cm[index], outer) - (domain.disk_area(xy_cm[index], inner) if inner else 0.))
        if part > 1e-9:
            pieces.append({"site_id": site_ids[index], "xy_cm": np.asarray(xy_cm[index]).tolist(),
                           "inner_radius_cm": inner, "outer_radius_cm": outer})
            area += part
    first = orbit[0]
    expected = (len(orbit) * math.pi * (outer_radius_cm[first]**2 - inner_radius_cm[first]**2)
                / domain.multiplier)
    if not pieces or not np.isclose(area, expected, rtol=1e-10, atol=1e-8):
        raise ArithmeticError("Sector annulus area differs from its physical multiplicity")
    return pieces, area


def channel_regions(domain, site_ids, xy_cm, interfaces_cm, z_edges_cm, orbits, material,
                    background, *, site_metadata=None, region_kind="cylinder"):
    """Build native cylindrical regions from explicit material and orbit data.

    Parameters
    ----------
    domain : HexagonalDomain
        Complete hexagon or one-sixth sector with its declared apothem.
    site_ids : sequence of str
        Stable physical cylinder-site identities.
    xy_cm : array_like
        Physical site x/y coordinates in cm, shaped (n_sites, 2).
    interfaces_cm : sequence of array_like
        Increasing positive shell-interface radii in cm for each physical site.
    z_edges_cm : array_like
        Increasing axial interval edges in cm.
    orbits : sequence of sequence of int
        Audited groups covering every physical site exactly once.
    material : callable
        Return material(site, shell, axial_level, full_volume_cm3), including atom_density
        and depletable fields.
    background : dict
        Material and identity fields for the surrounding axial background.
    site_metadata : sequence of dict or None, optional
        Additional per-site fields copied from each orbit's representative.
    region_kind : str, optional
        Caller-defined kind label stored on each cylinder region.

    Returns
    -------
    dict
        Native geometry, material regions, physical assembly-to-region map and
        volume-accounting checks.

    Raises
    ------
    ValueError
        Interfaces, orbit coverage or unique fuel identities are inconsistent.
    ArithmeticError
        Represented areas or total volumes fail closure.

    Notes
    -----
    material(site, shell, axial_level, full_volume_cm3) returns material fields, including
    depletable. The volume is the complete physical annulus at one site, before symmetry
    reduction. Each site has at most one depletable shell. Axial level zero is the top
    interval. Explicit orbits group equivalent physical sites; the caller audits their
    material symmetry and clearances.
    """
    regions, assembly_map, shapes = [], {}, []
    area_sum = 0.
    z = np.asarray(z_edges_cm)
    if any(np.any(np.diff(radii) <= 0) or np.min(radii) <= 0 for radii in interfaces_cm) or np.any(np.diff(z) <= 0):
        raise ValueError("Cylinder interfaces and axial edges must be strictly ordered")
    if sorted(i for orbit in orbits for i in orbit) != list(range(len(site_ids))):
        raise ValueError("Symmetry orbits must cover each physical site exactly once")
    for orbit in orbits:
        first = orbit[0]
        identity = min(site_ids[i] for i in orbit)
        if any(not np.array_equal(interfaces_cm[i], interfaces_cm[first]) for i in orbit):
            raise ValueError(f"Symmetry orbit has different cylinder interfaces: {identity}")
        visible = [i for i in orbit if domain.disk_area(xy_cm[i], interfaces_cm[i][-1]) > 1e-9]
        if not visible:
            raise ValueError(f"Orbit has no domain representative: {identity}")
        for shell, outer in enumerate(interfaces_cm[first]):
            inner = interfaces_cm[first][shell - 1] if shell else 0.
            pieces = [{"site_id": site_ids[i], "xy_cm": np.asarray(xy_cm[i]).tolist(),
                       "inner_radius_cm": float(inner), "outer_radius_cm": float(outer)} for i in visible]
            area = sum(math.pi * (outer * outer - inner * inner) if domain.full else
                       domain.disk_area(xy_cm[i], outer) - (domain.disk_area(xy_cm[i], inner) if inner else 0.)
                       for i in visible)
            expected = len(orbit) * math.pi * (outer * outer - inner * inner) / domain.multiplier
            if not np.isclose(area, expected, rtol=1e-10, atol=1e-8):
                raise ArithmeticError(f"Sector area/multiplicity disagreement for {identity}/{shell}")
            area_sum += area
            for slab, (lower, upper) in enumerate(zip(z[:-1], z[1:])):
                level = len(z) - 2 - slab
                volume = (math.pi * outer * outer if not inner else
                          math.pi * (outer * outer - inner * inner)) * (upper - lower)
                fields = material(first, shell, level, volume)
                fuel = fields["depletable"]
                label = f"{identity}/axial/{level}" if fuel else f"{identity}/shell/{shell}/axial/{level}"
                entry = {"label": label, "kind": region_kind, "pieces": pieces,
                         "orbit_site_ids": [site_ids[i] for i in orbit], "interface_index": shell,
                         "z_bounds_cm": [float(lower), float(upper)], "volume_cm3": float(area * (upper - lower)),
                         "physical_multiplicity": len(orbit), "void": not bool(fields["atom_density"]),
                         **({} if site_metadata is None else site_metadata[first]), **fields}
                if fuel:
                    entry.update(full_assembly_volume_cm3=volume,
                                 physical_assembly_ids=[f"{site_ids[i]}/axial/{level}" for i in orbit])
                    for physical_id in entry["physical_assembly_ids"]:
                        if physical_id in assembly_map:
                            raise ValueError(f"Duplicate fuel assembly identity: {physical_id}")
                        assembly_map[physical_id] = len(regions)
                regions.append(entry)
        shapes.extend({"site_id": site_ids[i], "xy_cm": np.asarray(xy_cm[i]).tolist(),
                       "outer_radius_cm": float(interfaces_cm[i][-1])} for i in visible)
    if area_sum >= domain.area:
        raise ArithmeticError("Cylinders leave no background material in the domain")
    for slab, (lower, upper) in enumerate(zip(z[:-1], z[1:])):
        regions.append({**background, "label": f"{background['label']}/axial/{len(z) - 2 - slab}",
                        "z_bounds_cm": [float(lower), float(upper)],
                        "volume_cm3": float((domain.area - area_sum) * (upper - lower))})
    total = domain.area * (z[-1] - z[0])
    if not np.isclose(sum(r["volume_cm3"] for r in regions), total, rtol=1e-12):
        raise ArithmeticError("Cylinder and background volumes do not cover the domain")
    lower, upper = domain.bounds(z)
    return {"geometry": {"symmetry": "full" if domain.full else "c6", "apothem_cm": domain.apothem_cm,
                         "lower_cm": lower, "upper_cm": upper, "channel_shapes": shapes},
            "regions": regions, "physical_assembly_to_region": assembly_map,
            "geometry_checks": {"volume_cm3": total, "orbits": len(orbits),
                                "independent_fuel_states": sum(r["depletable"] for r in regions),
                                "physical_axial_assemblies": len(assembly_map)}}


def insert_regions(case, additions, background_indices, exclusions):
    """Insert cylindrical regions while preserving existing state indices.

    Parameters
    ----------
    case : dict
        Native case to update in place.
    additions : sequence of dict
        New regions whose axial intervals exactly match existing background intervals.
    background_indices : sequence of int
        Indices of background regions displaced by the additions.
    exclusions : sequence of dict
        Cylinder shape records added to background geometric exclusions.

    Raises
    ------
    ValueError
        An addition uses an unsupported background interval.
    ArithmeticError
        The displacement removes all background or fails volume closure.

    Notes
    -----
    Split additions and exclusions at the existing background boundaries so native CSG
    exclusions and live material fractions use the same intervals.
    """
    bounds = {tuple(case["regions"][index]["z_bounds_cm"]) for index in background_indices}
    if any(tuple(region["z_bounds_cm"]) not in bounds for region in additions):
        raise ValueError("Added cylinders must use the existing background axial intervals")
    before = sum(region["volume_cm3"] for region in case["regions"])
    removed = []
    for index in background_indices:
        low, high = case["regions"][index]["z_bounds_cm"]
        displaced = sum(region["volume_cm3"] * max(0., min(high, region["z_bounds_cm"][1])
                                                  - max(low, region["z_bounds_cm"][0]))
                        / (region["z_bounds_cm"][1] - region["z_bounds_cm"][0]) for region in additions)
        if case["regions"][index]["volume_cm3"] <= displaced:
            raise ArithmeticError("Cylindrical additions displace the entire background")
        removed.append(displaced)
    added = sum(region["volume_cm3"] for region in additions)
    if not np.isclose(sum(removed), added, rtol=1e-12, atol=1e-10):
        raise ArithmeticError("Added cylinders do not match the selected background intervals")
    for index, amount in zip(background_indices, removed):
        case["regions"][index]["volume_cm3"] -= amount
    case["regions"].extend(additions)
    case["geometry"]["channel_shapes"].extend(exclusions)
    case["nuclides"] = sorted(set(case["nuclides"]) | {name for region in additions for name in region["atom_density"]})
    if not np.isclose(sum(region["volume_cm3"] for region in case["regions"]), before, rtol=1e-12):
        raise ArithmeticError("Adding cylinders changed the total domain volume")


@dataclass(frozen=True)
class Cylinder:
    """Live center and interfaces; any radius may map axial intervals to tensors."""
    xy_cm: torch.Tensor
    interfaces_cm: torch.Tensor | Sequence[torch.Tensor | Mapping[tuple[float, float], torch.Tensor]]


def cylinder_fractions(grid, regions, cylinders):
    """Compute live annular-cylinder fractions on a triangular-prism mesh.

    Parameters
    ----------
    grid : C6Grid
        Fixed triangular-prism mesh with vertices and axial edges in cm.
    regions : sequence of dict
        Native region descriptions in the requested material-column order.
    cylinders : mapping of str to Cylinder
        Live cylinder interfaces keyed by stable physical site identity.

    Returns
    -------
    torch.Tensor
        Cell-by-region volume fractions retaining geometry derivatives.

    Raises
    ------
    ArithmeticError
        Fractions are negative or fail to partition the represented domain.
    """
    areas = {}

    def disk(site, interface, bounds):
        """Cache a live circle-triangle intersection for one site interface.

        Parameters
        ----------
        site : str
            Physical cylinder identity.
        interface : int
            Radial interface index, allowing negative indexing.
        bounds : tuple of float
            Axial interval in cm selecting a height-dependent radius when applicable.

        Returns
        -------
        torch.Tensor
            Disk intersection area in cm² for each triangular cell.
        """
        cylinder = cylinders[site]
        interface %= len(cylinder.interfaces_cm)
        radius = cylinder.interfaces_cm[interface]
        axial = isinstance(radius, Mapping)
        key = (site, interface, bounds if axial else None)
        if key not in areas:
            areas[key] = circle_triangle_area(grid.vertices, cylinder.xy_cm, radius[bounds] if axial else radius)
        return areas[key]

    occupied = {}
    columns = []
    for region in regions:
        bounds = tuple(region["z_bounds_cm"])
        if "pieces" not in region:
            if bounds not in occupied:
                occupied[bounds] = sum(disk(site, -1, bounds) for site in cylinders)
            area = grid.triangle_area - occupied[bounds]
        else:
            shell = region["interface_index"]
            area = sum(disk(site, shell, bounds) - (disk(site, shell - 1, bounds) if shell else 0.)
                       for site in region["orbit_site_ids"])
        low, high = bounds
        overlap = (torch.minimum(grid.z_edges[1:], tensor(high))
                   - torch.maximum(grid.z_edges[:-1], tensor(low))).clamp_min(0.)
        columns.append(((area / grid.triangle_area)[:, None] * (overlap / grid.z_edges.diff())[None, :]).flatten())
    fractions = torch.stack(columns, dim=-1)
    if float(fractions.detach().min()) < -2e-10 or float((fractions.detach().sum(-1) - 1).abs().max()) > 2e-10:
        raise ArithmeticError("Cylinder material fractions do not partition the triangular domain")
    return fractions.clamp_min(0.)
