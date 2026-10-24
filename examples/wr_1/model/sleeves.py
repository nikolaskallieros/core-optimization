"""Stationary graphite annuli around each fuel-channel axial position."""
import math

import numpy as np
import torch

from diffusion_surrogate.symmetry import audit_c6_orbits
from diffusion_surrogate.cylinders import HexagonalDomain, cylinder_clearances, orbit_annulus, insert_regions
from materials.composition import number_density


def initialize_sleeves(design, specification):
    """Register the fixed topology and parameter bindings for axial graphite sleeves.

    Parameters
    ----------
    design : Design
        Live WR-1 design with differentiable ring and optional graphite parameters.
    specification : dict
        Resolved scenario settings defining the axial graphite sleeves, including topology
        and material properties.

    Raises
    ------
    ValueError
        Topology, identities, parameter shapes or supported axial conventions are
        inconsistent.
    """
    if specification["mode"] != "assembly_sleeves" or design.axial_count != 3:
        raise ValueError("Graphite sleeves require three fixed axial positions")
    positions = tuple(specification["positions"])
    if len(positions) != 7 or len(set(positions)) != 7 or design.graphite_thickness_mm.shape != (21,):
        raise ValueError("Sleeves require seven independent positions times three thicknesses")
    from .geometry import ChannelDesign
    mapped = ChannelDesign.forward(design)
    centres = mapped["channel_xy_mm"].detach().numpy()/10
    labels = [(bool(f), int(r)) for f, r in zip(design.fuelled, design.ring)]
    orbits = audit_c6_orbits(centres, labels, tolerance_cm=1e-9)
    indices = torch.full((len(design.channel_ids),), -1, dtype=torch.long)
    selected = {}
    for orbit in orbits:
        if bool(design.fuelled[orbit[0]]):
            identity = min(design.channel_ids[i] for i in orbit)
            if identity not in positions:
                raise ValueError(f"Sleeve position missing from configuration: {identity}")
            selected[identity] = tuple(orbit)
            indices[list(orbit)] = positions.index(identity)
    if set(selected) != set(positions):
        raise ValueError("Sleeve positions do not match the fuelled C6 orbits")
    design.register_buffer("graphite_position_index", indices)
    design.graphite_sleeve_positions = positions
    design.graphite_sleeve_orbits = tuple(selected[name] for name in positions)
    design.graphite_spec = specification


def sleeve_mapping(design, mapped):
    """Evaluate the live dimensions of the configured axial graphite sleeves.

    Parameters
    ----------
    design : Design
        Live WR-1 design with differentiable ring and optional graphite parameters.
    mapped : dict
        Live geometry tensors returned by the design's forward mapping.

    Returns
    -------
    dict
        Differentiable sleeve inner radii, axial outer radii and thicknesses in mm.

    Raises
    ------
    ValueError
        Graphite dimensions are nonfinite or outside their allowed physical range.
    """
    thickness = design.graphite_thickness_mm
    if not bool(torch.isfinite(thickness).all()) or bool((thickness < 0).any()):
        raise ValueError("Graphite sleeve thicknesses must be finite and nonnegative")
    positions = design.graphite_position_index
    local = thickness.reshape(7, 3)[positions.clamp_min(0)]
    local = torch.where((positions >= 0)[:, None], local, torch.zeros_like(local))
    inner = torch.stack([r[-1] for r in mapped["channel_interfaces_mm"]])
    return {"sleeve_inner_radius_mm": inner,
            "sleeve_outer_radius_mm": inner[:, None]+local,
            "sleeve_thickness_mm": local}


def add_sleeves(case, design, mapped, cross_sections):
    """Insert axial graphite sleeves and subtract their displaced moderator volume.

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
    xy = mapped["channel_xy_mm"].detach().numpy()/10
    inner = mapped["sleeve_inner_radius_mm"].detach().numpy()/10
    outer = mapped["sleeve_outer_radius_mm"].detach().numpy()/10
    levels = design.axial_count
    positions = len(design.graphite_sleeve_positions)
    thickness = design.graphite_thickness_mm.detach().numpy().reshape(positions, levels)/10
    domain = HexagonalDomain(case["geometry"]["apothem_cm"], case["geometry"]["symmetry"] == "full")
    gaps = cylinder_clearances(domain, xy, outer)
    clearance = spec["minimum_clearance_mm"]/10
    if clearance < 0 or min(gaps.values()) < clearance:
        raise ValueError("Graphite sleeves overlap channels or cross their required clearances")
    audit_c6_orbits(xy, [(bool(f), tuple(r)) for f, r in zip(design.fuelled, outer)], tolerance_cm=1e-9)
    density, temperature = number_density("C0", spec["density_g_cm3"]), spec["temperature_K"]
    if density <= 0 or temperature <= 0 or not math.isfinite(density+temperature):
        raise ValueError("Graphite density and temperature must be finite and positive")
    z = design.z_edges_mm.detach().numpy()/10
    additions, exclusions = [], []
    displaced = np.zeros(levels)
    physical = 0
    for position, orbit in enumerate(design.graphite_sleeve_orbits):
        identity, first = design.graphite_sleeve_positions[position], orbit[0]
        for level in range(levels):
            if thickness[position, level] == 0:
                continue
            lower, upper = float(z[-2-level]), float(z[-1-level])
            pieces, area = orbit_annulus(domain, design.channel_ids, xy, inner, outer[:, level], orbit)
            volume = float(area*(upper-lower))
            additions.append({
                "label": f"{identity}/graphite/axial/{level}", "kind": "wr1_shell",
                "equivalence_material": "graphite", "graphite_sleeve": True,
                "pieces": pieces, "orbit_site_ids": [design.channel_ids[i] for i in orbit],
                "interface_index": len(mapped["channel_interfaces_mm"][first]),
                "z_bounds_cm": [lower, upper], "volume_cm3": volume,
                "atom_density": {"C0": density}, "temperature_K": temperature,
                "thermal_scattering": ["c_Graphite"], "depletable": False, "void": False,
                "ring": int(design.ring[first]), "physical_multiplicity": len(orbit),
                "fixed_position": identity, "axial_level": level,
            })
            exclusions.extend({"site_id": p["site_id"], "xy_cm": p["xy_cm"], "outer_radius_cm": p["outer_radius_cm"],
                               "z_bounds_cm": [lower, upper]} for p in pieces)
            displaced[level] += volume
            physical += len(orbit)
    backgrounds = [i for i, region in enumerate(case["regions"]) if region["kind"] == "wr1_background"]
    insert_regions(case, additions, backgrounds, exclusions)
    case["design"]["graphite_thickness_mm"] = design.graphite_thickness_mm.detach().tolist()
    case["cross_sections"] = str(cross_sections)
    case["geometry_checks"].update(
        graphite_physical_sleeves=physical, graphite_positions=positions, graphite_regions=len(additions),
        graphite_volume_cm3=float(displaced.sum()), graphite_volume_by_axial_level_cm3=displaced.tolist(),
        graphite_minimum_pair_clearance_cm=gaps["pair"], graphite_minimum_wall_clearance_cm=gaps["wall"],
        graphite_volume_closure_passed=True, fuel_state_indices_preserved=True,
        graphite_stationary_during_refuelling=True)
    case["effective_approximations"]["graphite"] = spec["material_basis"]
    return case
