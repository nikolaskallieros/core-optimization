"""Independent CAD, native-cell, mesh-volume and shape-derivative checks."""

import math

import build123d as bd
import numpy as np
import openmc
import pytest
import torch

from diffusion_surrogate import Diffusion3D
from geometry import CadGeometry, MaterialVolume, box, cylinder, union
from geometry.channels import hexagonal_domain
from geometry.diffusion import discretize
from geometry.openmc import from_cad


def pin_geometry(radius, center=None):
    """Construct an independent pin in a rectangular moderator volume.

    Parameters
    ----------
    radius : torch.Tensor
        Live fuel radius in cm.
    center : torch.Tensor or None, optional
        Live x/y center in cm; None uses a fixed off-center position.

    Returns
    -------
    CadGeometry
        Two material volumes with vacuum radial and reflective axial boundaries.
    """
    center = radius.new_tensor([1.3, .85]) if center is None else center
    domain = box([0., 0., 0.], [3., 2., 1.])
    fuel = cylinder(center, radius, [0., 1.])
    faces = ("x_min", "x_max", "y_min", "y_max", "bottom", "top")
    boundaries = {name: {"type": "vacuum" if i < 4 else "reflective", "slot": i}
                  for i, name in enumerate(faces)}
    return CadGeometry(domain, (MaterialVolume("fuel", "fuel", fuel),
                               MaterialVolume("moderator", "moderator", domain - fuel)), boundaries, "pin")


def test_step_solids_native_cells_and_diffusion_volumes_agree(tmp_path):
    """Check actual STEP BREP identities, units, ownership and exact volumes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated STEP output folder supplied by pytest.
    """
    radius = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    cad = pin_geometry(radius)
    path = cad.write_step(tmp_path / "geometry.step")
    imported = CadGeometry.read_step(path)
    assert imported.is_direct
    assert [volume.label for volume in imported.volumes] == ["fuel", "moderator"]
    expected = np.array([math.pi * .4**2, 6. - math.pi * .4**2])
    np.testing.assert_allclose([volume.shape.volume / 1000 for volume in imported.volumes], expected, rtol=1e-11)
    fixed = discretize(cad, (7, 5, 3))
    np.testing.assert_allclose((fixed.fractions().T @ fixed.grid.volumes).detach().numpy(), expected, rtol=1e-11)
    openmc.reset_auto_ids()
    materials = {name: openmc.Material(name=name) for name in ("fuel", "moderator")}
    native = from_cad(path, materials)
    for point, label in (((1.3, .85, .5), "fuel"), ((2.7, 1.7, .5), "moderator")):
        owners = [cell for cell in native.cells if point in cell.region]
        assert len(owners) == 1 and owners[0].fill.name == label
        solids = [volume for volume in imported.volumes
                  if any(solid.is_inside(tuple(10 * value for value in point)) for solid in volume.shape.solids())]
        assert [volume.label for volume in solids] == [label]
    with torch.no_grad():
        radius.copy_(radius.new_tensor(.47))
    cad.write_step(path)
    assert list(tmp_path.iterdir()) == [path]
    assert bd.import_step(path).children[0].volume / 1000 == pytest.approx(math.pi * .47**2)


def test_gmsh_sector_preserves_material_volumes_and_periodic_planes(tmp_path):
    """Compare an exact one-sixth cylinder against the CAD-derived sector mesh.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated STEP output folder supplied by pytest.
    """
    radius = torch.tensor(.6, dtype=torch.float64, requires_grad=True)
    domain, boundaries = hexagonal_domain({"apothem_cm": 3., "symmetry": "c6",
                                           "lower_cm": [0., -2., 0.], "upper_cm": [3., 2., 2.]})
    fuel = cylinder([0., 0.], radius, [0., 2.])
    cad = CadGeometry(domain, (MaterialVolume("fuel", "fuel", fuel),
                               MaterialVolume("moderator", "moderator", domain - fuel)), boundaries)
    mesh = discretize(cad, 6, axial_subdivisions=2)
    volume = (mesh.fractions().T @ mesh.grid.volumes)[0]
    assert float(volume.detach()) == pytest.approx(math.pi * .6**2 * 2 / 6, rel=1e-11)
    derivative, = torch.autograd.grad(volume, radius)
    assert float(derivative) == pytest.approx(2 * math.pi * .6 * 2 / 6, rel=1e-11)
    assert len(mesh.grid.periodic_pairs) == 5
    step = cad.write_step(tmp_path / "geometry.step")
    assert bd.import_step(step).children[0].volume / 1000 == pytest.approx(float(volume.detach()))
    openmc.reset_auto_ids()
    native = from_cad(step, {"fuel": None, "moderator": None})
    lower, upper = native.boundaries["side/0"], native.boundaries["side/2"]
    assert lower.periodic_surface is upper
    assert lower.evaluate((1., 0., 1.)) > 0 and upper.evaluate((1., 0., 1.)) > 0


def test_manufactured_diffusion_shape_gradients_match_independent_differences():
    """Check one backward pass for the radius and both center coordinates."""
    values = torch.tensor([.4, 1.3, .85], dtype=torch.float64, requires_grad=True)
    absorption = torch.tensor([[.08], [.01]], dtype=torch.float64)
    fission = torch.tensor([[.06], [0.]], dtype=torch.float64)
    scatter = torch.tensor([[[.2]], [[1.]]], dtype=torch.float64)
    macro = {"absorption": absorption, "fission": fission, "nu_fission": 2.5 * fission,
             "kappa_fission": 200e6 * fission, "scatter": scatter, "nu_scatter": scatter,
             "transport": absorption + scatter.sum(-1), "chi": torch.ones_like(fission)}

    def evaluate(parameters):
        """Solve the same CAD-defined pin with manufactured material data.

        Parameters
        ----------
        parameters : torch.Tensor
            Radius and x/y center coordinates in cm.

        Returns
        -------
        torch.Tensor
            Effective multiplication factor from the shared diffusion solver.
        """
        mesh = discretize(pin_geometry(parameters[0], parameters[1:]), (5, 4, 2))
        return Diffusion3D(mesh.grid).solve(mesh.fractions(), macro)["keff"]

    gradient, = torch.autograd.grad(evaluate(values), values)
    differences = []
    for axis in range(3):
        delta = torch.zeros_like(values)
        delta[axis] = 1e-5
        differences.append((evaluate(values.detach() + delta) - evaluate(values.detach() - delta)) / 2e-5)
    torch.testing.assert_close(gradient, torch.stack(differences), rtol=2e-5, atol=1e-8)
    assert bool((gradient.abs() > 1e-5).all())


def test_boolean_union_does_not_double_count_overlap():
    """Verify inclusion-exclusion and a moving rectangular interface."""
    radius = torch.tensor(.4, dtype=torch.float64)
    base = pin_geometry(radius)
    width = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    first = box([0., 0., 0.], [width, 2., 1.])
    second = box([.5, 0., 0.], [2., 1., 1.])
    combined = union((first, second))
    cad = CadGeometry(base.domain, (MaterialVolume("combined", "combined", combined),
                                   MaterialVolume("rest", "rest", base.domain - combined)), base.boundaries)
    mesh = discretize(cad, (7, 5, 2))
    volumes = mesh.fractions().T @ mesh.grid.volumes
    torch.testing.assert_close(volumes, torch.stack((width + 2., 4. - width)), rtol=1e-11, atol=1e-12)
    derivative, = torch.autograd.grad(volumes[0], width)
    assert float(derivative) == pytest.approx(1., abs=1e-11)


def test_overlapping_material_regions_fail_at_the_mapping_boundary():
    """Reject a non-partition instead of normalizing away a physical error."""
    base = pin_geometry(torch.tensor(.4, dtype=torch.float64))
    cad = CadGeometry(base.domain, (MaterialVolume("a", "a", base.domain),
                                   MaterialVolume("b", "b", base.domain)), base.boundaries)
    with pytest.raises(ArithmeticError, match="partition"):
        discretize(cad, (2, 2, 1)).fractions()


def test_native_material_bindings_allow_independent_volume_histories():
    """Keep local compositions distinct when two volumes share a material family."""
    base = pin_geometry(torch.tensor(.4, dtype=torch.float64))
    first = base.domain.slice([0., .5])
    second = base.domain.slice([.5, 1.])
    cad = CadGeometry(base.domain, (MaterialVolume("lower", "fuel", first),
                                   MaterialVolume("upper", "fuel", second)), base.boundaries)
    openmc.reset_auto_ids()
    fresh, depleted = openmc.Material(name="fresh"), openmc.Material(name="depleted")
    native = from_cad(cad, {"fuel": fresh, "upper": depleted})
    assert [cell.fill.name for cell in native.cells] == ["fresh", "depleted"]
