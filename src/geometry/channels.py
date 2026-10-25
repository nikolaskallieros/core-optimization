"""Material-region catalogues and shared CAD constructions for axial channels."""
from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .intersections import circle_triangle_area
from .model import CadGeometry, MaterialVolume, cylinder, prism, union


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
        return math.pi * radius * radius if self.full else float(circle_triangle_area(torch.as_tensor(self.polygon, dtype=torch.float64)[None],
                                         torch.as_tensor(xy, dtype=torch.float64),
                                         torch.as_tensor(radius, dtype=torch.float64))[0])

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


def hexagonal_domain(specification):
    """Construct a CAD hexagon or rotational sector with named boundaries.

    Parameters
    ----------
    specification : dict
        Apothem, lower/upper coordinates in cm, symmetry, six diffusion boundary
        types and optional symmetry_boundary_type.

    Returns
    -------
    tuple
        Shared CAD prism and its domain-face boundary specifications.
    """
    a = specification["apothem_cm"]
    h = a / math.sqrt(3)
    full = specification["symmetry"] == "full"
    vertices = ([[a, -h], [a, h], [0., 2*h], [-a, h], [-a, -h], [0., -2*h]] if full else
                [[0., 0.], [a, -h], [a, h]])
    domain = prism(vertices, [specification["lower_cm"][2], specification["upper_cm"][2]])
    physical = specification.get("boundaries", ["vacuum"] * 6)
    boundaries = {f"side/{i}": {"type": physical[0], "slot": 0} for i in range(len(vertices))}
    if not full:
        symmetry = specification.get("symmetry_boundary_type", "periodic")
        for i, other in ((0, 2), (2, 0)):
            boundaries[f"side/{i}"] = {"type": symmetry, "pair": f"side/{other}"}
    boundaries.update(bottom={"type": physical[4], "slot": 4}, top={"type": physical[5], "slot": 5})
    return domain, boundaries


def channel_geometry(domain, regions, cylinders, boundaries, *, name="channels"):
    """Build one parametric CAD assembly for annular channels and their background.

    Parameters
    ----------
    domain : Shape
        Convex polygonal CAD prism enclosing the material volumes.
    regions : sequence of dict
        Ordered material identities and axial intervals. Channel regions provide
        orbit_site_ids and interface_index; other regions fill the background.
        Numerical coordinates in generated region records are not read here.
    cylinders : mapping of str to Cylinder
        Live channel centers and radial interfaces, keyed by physical site.
        An interface may map axial bounds to a local radius, for example a sleeve.
    boundaries : dict
        Domain-face boundary conditions shared by both numerical adapters.
    name : str, optional
        CAD assembly name.

    Returns
    -------
    CadGeometry
        Build123d construction with one material volume per region. Native cells
        and diffusion fractions both derive from this object.
    """
    disks, visible = {}, {}
    z = domain.parameters["z"]

    def disk(site, interface, bounds):
        """Reuse a live full-height radial interface across axial regions.

        Parameters
        ----------
        site : str
            Physical channel identity.
        interface : int
            Radial interface index, including negative indexing.
        bounds : tuple of float
            Axial interval selecting a height-dependent interface.

        Returns
        -------
        Shape
            Shared cylinder construction.
        """
        channel = cylinders[site]
        interface %= len(channel.interfaces_cm)
        radius = channel.interfaces_cm[interface]
        axial = isinstance(radius, Mapping)
        key = (site, interface, bounds if axial else None)
        if key not in disks:
            disks[key] = cylinder(channel.xy_cm, radius[bounds] if axial else radius, z)
        return disks[key]

    def represented(site, bounds):
        """Select physical cylinders intersecting the current symmetry domain.

        Parameters
        ----------
        site : str
            Physical channel identity.
        bounds : tuple of float
            Axial interval for the outer interface.

        Returns
        -------
        bool
            Whether the represented outer cylinder has nonzero transverse area.
        """
        outer = disk(site, -1, bounds)
        if outer not in visible:
            p = outer.parameters
            area = circle_triangle_area(domain.parameters["vertices"][None], p["xy"], p["radius"])
            visible[outer] = bool(area.detach()[0] > 1e-9)
        return visible[outer]

    volumes = []
    for region in regions:
        bounds = tuple(region["z_bounds_cm"])
        if "pieces" in region:
            shell = region["interface_index"]
            shapes = [disk(site, shell, bounds) - disk(site, shell - 1, bounds) if shell else disk(site, shell, bounds)
                      for site in region["orbit_site_ids"] if represented(site, bounds)]
            shape = union(shapes)
        else:
            shape = domain - union(disk(site, -1, bounds) for site in cylinders if represented(site, bounds))
        volumes.append(MaterialVolume(region["label"], region["label"], (domain & shape).slice(bounds)))
    return CadGeometry(domain, tuple(volumes), boundaries, name)



class ChannelDesign(nn.Module):
    """Map declared hexagonal lattice sites and ring parameters to axial assemblies."""
    def __init__(self, definition, specification):
        """Bind resolved model definitions to trainable channel geometry.

        Parameters
        ----------
        definition : dict
            Resolved physical model from ``inputs/model.toml``, supplied by the case adapter.
        specification : dict
            Resolved cycle settings from ``inputs/study.toml`` after scenario overrides.

        Notes
        -----
        The case adapter loads model.toml and study.toml once and passes their resolved
        dictionaries here. This constructor registers design parameters and fixed lattice
        buffers without reading files.
        """
        super().__init__()
        for name, value in specification["design"].items():
            self.register_parameter(name, nn.Parameter(torch.tensor(value, dtype=torch.float64)))
        sites = definition["lattice"]["sites"]
        channel_types = definition["channel_types"]
        channels = [channel_types[site["channel_type"]] for site in sites]
        self.channel_ids = tuple(site["id"] for site in sites)
        rings = [max(abs(site["q"]), abs(site["r"]), abs(site["q"] + site["r"])) for site in sites]
        self.register_buffer("ring", torch.tensor(rings, dtype=torch.long))
        self.register_buffer("fuelled", torch.tensor([channel["fuelled"] for channel in channels], dtype=torch.bool))
        self.register_buffer("lattice_xy", torch.tensor([
            [site["q"] + site["r"] / 2, math.sqrt(3) * site["r"] / 2] for site in sites], dtype=torch.float64))
        self.axial_count = specification["axial_positions"]
        self.register_buffer("z_edges_mm", torch.arange(self.axial_count + 1, dtype=torch.float64)
                             * specification["assembly_height_mm"])
        self.reference_interfaces = tuple(tuple(channel["reference_interfaces_mm"]) for channel in channels)
        self.channel_materials = tuple(tuple(channel["materials"]) for channel in channels)
        self.assembly_ids = tuple(f"{site}/axial/{level}" for site, fuelled in zip(self.channel_ids, self.fuelled)
                                  if fuelled for level in range(self.axial_count))

    def ring_pitches(self):
        """Return the live radial ring-spacing increments.

        Returns
        -------
        torch.Tensor
            Ring-spacing increments in mm, in outward ring order.
        """
        return self.pitch_mm

    def forward(self):
        """Map live ring parameters to channel and axial assembly geometry.

        Returns
        -------
        dict
            Channel coordinates and interfaces in mm; assembly coordinates, radii and axial
            bounds in cm; assembly volumes in cm³ and fresh enrichment mass fractions. Axial
            assembly identities run top to bottom.

        Raises
        ------
        ValueError
            A pitch, radius or enrichment is nonfinite or outside its physical range.
        """
        if (not bool(torch.isfinite(self.pitch_mm).all() & torch.isfinite(self.element_radius_mm).all()
                     & torch.isfinite(self.enrichment).all())
                or bool((self.pitch_mm <= 0).any() | (self.element_radius_mm <= 0).any()
                        | (self.enrichment <= 0).any() | (self.enrichment >= 1).any())):
            raise ValueError("Invalid ring pitch, homogenized element radius or enrichment")
        pitches = self.ring_pitches()
        scales = torch.cat((pitches.new_zeros(1), pitches.cumsum(0)
                            / torch.arange(1, len(pitches) + 1, dtype=pitches.dtype,
                                           device=pitches.device)))
        centers_mm = self.lattice_xy * scales[self.ring, None]
        interfaces_mm = []
        for i, values in enumerate(self.reference_interfaces):
            radii = self.pitch_mm.new_tensor(values)
            if bool(self.fuelled[i]):
                radii = radii + self.element_radius_mm[self.ring[i]] - radii[0]
            interfaces_mm.append(radii)
        fuel_ring = self.ring[self.fuelled].repeat_interleave(self.axial_count)
        # Assembly zero is at the top; physical z increases bottom-to-top.
        bounds_cm = torch.stack((self.z_edges_mm[:-1], self.z_edges_mm[1:]), dim=1).flip(0) / 10
        bounds_cm = bounds_cm.repeat(int(self.fuelled.sum()), 1)
        radius_cm = self.element_radius_mm[fuel_ring] / 10
        return {
            "channel_xy_mm": centers_mm,
            "channel_interfaces_mm": tuple(interfaces_mm),
            "assembly_xy_cm": centers_mm[self.fuelled].repeat_interleave(self.axial_count, 0) / 10,
            "assembly_radius_cm": radius_cm,
            "assembly_z_bounds_cm": bounds_cm,
            "assembly_volume_cm3": math.pi * radius_cm.square() * (bounds_cm[:, 1] - bounds_cm[:, 0]),
            "fresh_enrichment_mass_fraction": self.enrichment[fuel_ring],
        }
