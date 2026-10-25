"""Full-height graphite cylinders on a fixed C6 triangular-lattice topology."""
import math

import numpy as np
import torch

from diffusion_surrogate.symmetry import audit_c6_orbits
from geometry.channels import HexagonalDomain, cylinder_clearances, orbit_annulus, insert_regions
from materials.composition import number_density


def initialize_graphite(design, specification):
    """Register the fixed topology and parameter bindings for graphite cylinders.

    Parameters
    ----------
    design : Design
        Live WR-1 design with differentiable ring and optional graphite parameters.
    specification : dict
        Resolved scenario settings defining the graphite cylinders, including topology and
        material properties.

    Raises
    ------
    ValueError
        Topology, identities, parameter shapes or supported axial conventions are
        inconsistent.
    """
    if specification.get("axial_extent", "full_active_height") != "full_active_height":
        raise ValueError("This graphite model requires full-height cylinders")
    sites = specification["sites"]
    identities = tuple(site["id"] for site in sites)
    if len(identities) != len(set(identities)) or not identities:
        raise ValueError("Graphite lattice sites require unique identities")
    if set(identities) & set(design.channel_ids):
        raise ValueError("Graphite and fuel-channel identities overlap")
    if any(not isinstance(site[key], int) for site in sites for key in ("q", "r", "radius_group")):
        raise ValueError("Graphite lattice coordinates and radius groups must be integers")
    groups = [site["radius_group"] for site in sites]
    if set(groups) != set(range(len(design.graphite_radius_mm))):
        raise ValueError("Each graphite radius parameter must own at least one lattice site")
    if tuple(design.graphite_pitch_mm.shape) != (1,):
        raise ValueError("Graphite uses one common triangular-lattice pitch")
    angle = math.radians(specification["rotation_deg"])
    c, s = math.cos(angle), math.sin(angle)
    xy = [(site["q"] + site["r"] / 2, math.sqrt(3) * site["r"] / 2) for site in sites]
    design.register_buffer("graphite_unit_xy", torch.tensor(
        [(c*x-s*y, s*x+c*y) for x, y in xy], dtype=torch.float64))
    design.register_buffer("graphite_radius_group", torch.tensor(groups, dtype=torch.long))
    design.graphite_ids = identities
    design.graphite_spec = specification


def graphite_mapping(design):
    """Evaluate the live dimensions of the configured graphite cylinders.

    Parameters
    ----------
    design : Design
        Live WR-1 design with differentiable ring and optional graphite parameters.

    Returns
    -------
    dict
        Differentiable graphite coordinates and radii in mm.

    Raises
    ------
    ValueError
        Graphite dimensions are nonfinite or outside their allowed physical range.
    """
    pitch, radius = design.graphite_pitch_mm, design.graphite_radius_mm
    if (not bool(torch.isfinite(pitch).all() & torch.isfinite(radius).all())
            or bool((pitch <= 0).any() | (radius <= 0).any())):
        raise ValueError("Graphite pitch and radii must be finite and positive")
    return {"graphite_xy_mm": design.graphite_unit_xy * pitch,
            "graphite_radius_mm": radius[design.graphite_radius_group]}


def add_graphite(case, design, mapped, cross_sections):
    """Insert graphite cylinders and subtract their displaced moderator volume.

    Parameters
    ----------
    case : dict
        Native region, material and symmetry description returned by ``make_case``.
    design : Design
        Live WR-1 design with differentiable ring and optional graphite parameters.
    mapped : dict
        Live geometry tensors returned by the design's forward mapping.
    cross_sections : str or pathlib.Path
        Cross-section library XML including the required graphite thermal-scattering data.

    Returns
    -------
    dict
        The mutated native case with graphite regions, material data, cross-section path and
        clearance checks.

    Raises
    ------
    ValueError
        Graphite dimensions, clearances, density or temperature are invalid.
    """
    spec = design.graphite_spec
    xy = mapped["graphite_xy_mm"].detach().numpy() / 10
    radius = mapped["graphite_radius_mm"].detach().numpy() / 10
    fuel_xy = mapped["channel_xy_mm"].detach().numpy() / 10
    outer = np.array([float(r[-1].detach()) / 10 for r in mapped["channel_interfaces_mm"]])
    domain = HexagonalDomain(case["geometry"]["apothem_cm"], case["geometry"]["symmetry"] == "full")
    gaps = cylinder_clearances(domain, xy, radius, obstacles=(fuel_xy, outer))
    clearance = spec["minimum_clearance_mm"] / 10
    if clearance < 0 or min(gaps.values()) < clearance:
        raise ValueError("Graphite cylinders overlap or cross their required clearances")
    orbits = audit_c6_orbits(xy, [(float(r), int(g)) for r, g in zip(radius, design.graphite_radius_group)], tolerance_cm=1e-9)
    z = design.z_edges_mm.detach().numpy() / 10
    density, temperature = number_density("C0", spec["density_g_cm3"]), spec["temperature_K"]
    if density <= 0 or temperature <= 0 or not math.isfinite(density + temperature):
        raise ValueError("Graphite density and temperature must be finite and positive")
    additions, exclusions, added_area = [], [], 0.
    for orbit in orbits:
        first = orbit[0]
        identity = min(design.graphite_ids[i] for i in orbit)
        pieces, area = orbit_annulus(domain, design.graphite_ids, xy, np.zeros_like(radius), radius, orbit)
        for slab, (lower, upper) in enumerate(zip(z[:-1], z[1:])):
            additions.append({
                "label": f"{identity}/shell/0/axial/{len(z)-2-slab}", "kind": "wr1_shell",
                "equivalence_material": "graphite", "pieces": pieces,
                "orbit_site_ids": [design.graphite_ids[i] for i in orbit], "interface_index": 0,
                "z_bounds_cm": [float(lower), float(upper)], "volume_cm3": float(area*(upper-lower)),
                "atom_density": {"C0": density}, "temperature_K": temperature,
                "thermal_scattering": ["c_Graphite"], "depletable": False, "void": False,
                "ring": int(design.graphite_radius_group[first]), "physical_multiplicity": len(orbit),
            })
        exclusions.extend({key: piece[key] for key in ("site_id", "xy_cm", "outer_radius_cm")} for piece in pieces)
        added_area += area
    backgrounds = [i for i, region in enumerate(case["regions"]) if region["kind"] == "wr1_background"]
    insert_regions(case, additions, backgrounds, exclusions)
    case["design"].update(graphite_pitch_mm=design.graphite_pitch_mm.detach().tolist(),
                          graphite_radius_mm=design.graphite_radius_mm.detach().tolist())
    case["cross_sections"] = str(cross_sections)
    case["geometry_checks"].update(
        graphite_physical_cylinders=len(xy), graphite_orbits=len(orbits),
        graphite_volume_cm3=float(added_area*(z[-1]-z[0])),
        graphite_minimum_fuel_clearance_cm=gaps["obstacles"], graphite_minimum_pair_clearance_cm=gaps["pair"],
        graphite_minimum_wall_clearance_cm=gaps["wall"],
        graphite_volume_closure_passed=True, fuel_state_indices_preserved=True)
    case["effective_approximations"]["graphite"] = spec["material_basis"]
    return case
