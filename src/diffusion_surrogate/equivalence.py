"""Spatial equivalence bases with caller-defined groups and fixed anchors."""
import torch
from diffusion_surrogate.solver import tensor


def equivalence_bases(assignments, groups, *, axial_coordinates=None, axial_mask=None):
    """Construct regional flux factors with fixed amplitude anchors.

    Parameters
    ----------
    assignments : sequence of str or None
        One factor-group label per region; None fixes that region's amplitude at unity.
    groups : int
        Number of energy groups in the native and surrogate tables.
    axial_coordinates : array_like or None, optional
        Normalized axial coordinate per region; enables linear and quadratic axial factors
        when supplied.
    axial_mask : array_like or None, optional
        Region mask applied to axial modes, for example to exclude void regions.

    Returns
    -------
    tuple
        Coefficient names, a diffusion-basis mapping and a coefficient-by-region-by-group
        equivalence basis.

    Notes
    -----
    Each named category shares one factor per energy group. Optional normalized axial
    coordinates add linear and quadratic factors. Diffusion has one factor per energy group;
    the vacuum extrapolation remains fixed.
    """
    categories = sorted({name for name in assignments if name is not None})
    names = ([f"diffusion_g{g}" for g in range(groups)]
             + [f"equivalence:{name}:g{g}" for name in categories for g in range(groups)])
    if axial_coordinates is not None:
        names += [f"axial_{mode}:g{g}" for mode in ("linear", "quadratic") for g in range(groups)]
    diffusion = torch.zeros((len(names), len(assignments), groups), dtype=torch.float64)
    equivalence = torch.zeros_like(diffusion)
    for g in range(groups):
        diffusion[g, :, g] = 1.
    for i, name in enumerate(categories):
        selected = [j for j, label in enumerate(assignments) if label == name]
        for g in range(groups):
            equivalence[groups + i * groups + g, selected, g] = 1.
    if axial_coordinates is not None:
        coordinate = tensor(axial_coordinates)
        offset = groups * (1 + len(categories))
        for mode, values in enumerate((coordinate, coordinate.square())):
            if axial_mask is not None:
                values = values * tensor(axial_mask)
            for g in range(groups):
                equivalence[offset + mode * groups + g, :, g] = values
    return names, {"diffusion": diffusion}, equivalence


def shared_bases(region_count, groups, scattering_regions, selected_group=0):
    """Construct seven shared correction modes for caller-selected regions.

    Parameters
    ----------
    region_count : int
        Number of material regions.
    groups : int
        Number of energy groups in the native and surrogate tables.
    scattering_regions : sequence of int
        Regions sharing the selected scattering correction.
    selected_group : int, optional
        Energy-group index for the group-specific absorption and diffusion modes.

    Returns
    -------
    tuple
        Seven coefficient names and their fission, capture, scattering, diffusion and
        boundary bases.

    Notes
    -----
    Other models may provide completely different bases to CalibratedDiffusion.
    """
    names = ('fission', 'selected_group_absorption', 'capture', 'selected_regions_scattering',
             'diffusion', 'selected_group_diffusion', 'vacuum_extrapolation')
    fission = torch.zeros((7, region_count, groups), dtype=torch.float64)
    capture, scatter, diffusion = (torch.zeros_like(fission) for _ in range(3))
    boundary = torch.zeros((7, 6, groups), dtype=torch.float64)
    fission[0], capture[2], diffusion[4], boundary[6] = 1., 1., 1., 1.
    fission[1, :, selected_group], capture[1, :, selected_group] = 1., 1.
    scatter[3, scattering_regions, :] = 1.
    diffusion[5, :, selected_group] = 1.
    return names, {'fission': fission, 'capture': capture, 'scatter': scatter,
                   'diffusion': diffusion, 'boundary': boundary}
