"""Physical inventory and shared-geometry checks for the zoned enrichment case."""

import math

import pytest
import torch

from designs.enrichment_shaping.build_geometry import build_geometry
from designs.enrichment_shaping.model.case import make_mesh, prepare, settings
from designs.enrichment_shaping.model.design import Enrichment, LatticeSize
from geometry.diffusion import discretize, subdivide_intervals
from optimization.objectives import relative_power_density, smooth_maximum


def test_490_segments_preserve_physical_volumes_and_mass_derivatives():
    """Recover cylinder volumes and independent finite differences of U235 mass."""
    config = settings()
    cad, mesh, fractions, enrichment, case = prepare(config)
    volumes = fractions.T @ mesh.grid.volumes
    expected = math.pi * config["geometry"]["fuel_radius_cm"]**2 * config["geometry"]["fuel_height_cm"] / 10
    torch.testing.assert_close(volumes[:-1], torch.full((490,), expected, dtype=torch.float64), rtol=1e-11, atol=1e-11)
    assert len(cad.volumes) == 491
    assert tuple(volume.label for volume in cad.volumes) == tuple(region["label"] for region in case["regions"])
    design = Enrichment(config, enrichment, volumes[:-1])
    mass = design.u235_mass_kg()
    derivative, = torch.autograd.grad(mass, design.enrichment_wt_percent)
    point = design.enrichment_wt_percent.detach().clone()
    step = 1e-4
    with torch.no_grad():
        design.enrichment_wt_percent[3, 7] += step
        upper = design.u235_mass_kg()
        design.enrichment_wt_percent[3, 7] -= 2 * step
        lower = design.u235_mass_kg()
        design.enrichment_wt_percent.copy_(point)
    torch.testing.assert_close(derivative[3, 7], (upper - lower) / (2 * step), rtol=2e-8, atol=1e-12)
    assert design.enrichment_wt_percent.numel() == 490
    assert case["need_depletion_rates"] is False


def test_mesh_edges_keep_interfaces_and_reject_a_cropped_physical_domain():
    """A faster background grid cannot silently remove part of the CAD domain."""
    physical = torch.tensor([0., 1., 4.], dtype=torch.float64, requires_grad=True)
    edges = subdivide_intervals(physical, [2, 3])
    torch.testing.assert_close(edges.detach(), torch.tensor([0., .5, 1., 2., 3., 4.], dtype=torch.float64))
    derivative, = torch.autograd.grad(edges[3], physical)
    torch.testing.assert_close(derivative, torch.tensor([0., 2/3, 1/3], dtype=torch.float64))
    cad, mesh, _, _, _ = prepare(settings())
    cropped = [value.clone() for value in mesh.grid.edges]
    cropped[0][0] += .1
    with pytest.raises(ValueError, match="endpoints differ"):
        discretize(cad, cartesian_edges=cropped)


def test_power_objective_uses_density_and_physical_volume_weighting():
    """Unequal segment sizes cannot hide a hotter local power density."""
    power = torch.tensor([10., 10.], dtype=torch.float64, requires_grad=True)
    volumes = torch.tensor([1., 3.], dtype=torch.float64)
    relative = relative_power_density(power, volumes)
    torch.testing.assert_close(relative, torch.tensor([2., 2/3], dtype=torch.float64))
    loss = smooth_maximum(relative, 50.)
    assert float(loss.detach()) == pytest.approx(2., abs=1e-12)
    gradient, = torch.autograd.grad(loss, power)
    torch.testing.assert_close(gradient, torch.tensor([.1, -.1], dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_transverse_sizing_preserves_material_cuts_and_the_exact_volume_derivative():
    """Check CAD-derived fraction invariance and dV/ds = 2 V/s independently."""
    design = LatticeSize(settings(), 3.)
    config = design.configuration()
    mesh = make_mesh(config, build_geometry(config["geometry"]))
    fractions = mesh.fractions()
    volume = (fractions.T @ mesh.grid.volumes)[:-1].sum()
    derivative, = torch.autograd.grad(volume, design.radial_scale)
    torch.testing.assert_close(derivative, 2 * volume.detach() / design.radial_scale.detach())
    with torch.no_grad():
        design.radial_scale *= 1.1
        changed = design.configuration()
        second = make_mesh(changed, build_geometry(changed["geometry"]))
        torch.testing.assert_close(second.fractions(), fractions.detach(), rtol=1e-10, atol=1e-12)
        new_volume = (second.fractions().T @ second.grid.volumes)[:-1].sum()
        torch.testing.assert_close(new_volume, volume.detach() * 1.1**2)
