"""Independent checks for direct CAD, unstructured diffusion and CAD transport."""

import sys
from pathlib import Path

import build123d as bd
import numpy as np
import openmc
import pytest
import torch
from scipy.optimize import brentq

from geometry import CadGeometry
from geometry.diffusion import discretize
from geometry.meshing import tetrahedral_mesh
from geometry.openmc import from_cad
from diffusion_surrogate.finite_element import FiniteElementDiffusion
from diffusion_surrogate import CalibratedDiffusion


def macro(absorption=.2):
    """Return manufactured one-group data for an independent solver check.

    Parameters
    ----------
    absorption : float or torch.Tensor, optional
        Macroscopic absorption in cm⁻¹.

    Returns
    -------
    dict
        Homogeneous one-region reaction tables with infinite-medium keff 1.25.
    """
    a = torch.as_tensor(absorption, dtype=torch.float64).reshape(1, 1)
    return {"absorption": a, "transport": a.new_tensor([[1.]]),
            "scatter": a.new_tensor([[[.8]]]), "nu_scatter": a.new_tensor([[[.8]]]),
            "fission": a.new_tensor([[.1]]), "nu_fission": a.new_tensor([[.25]]),
            "kappa_fission": a.new_tensor([[2e7]]), "chi": torch.ones_like(a)}


@pytest.fixture(scope="module")
def curved_mesh():
    """Mesh a tilted torus and the nonconvex surrounding material.

    Returns
    -------
    CadMesh
        Two conformal regions inside a reflective sphere.
    """
    torus = bd.Solid.make_torus(12, 3).rotate(bd.Axis.X, 25)
    sphere = bd.Solid.make_sphere(22)
    cad = CadGeometry.from_solids({"fuel": torus, "surroundings": sphere - torus},
        boundaries={"outside": {"type": "reflective", "faces": tuple(sphere.faces())}})
    return tetrahedral_mesh(cad, max_size_cm=.4, curvature_segments=36)


def test_curved_material_partition_and_step(curved_mesh, tmp_path):
    """Check actual BREP round-trip, material ownership and shared interfaces."""
    mesh = curved_mesh
    path = mesh.cad.write_step(tmp_path / "geometry.step")
    restored = CadGeometry.read_step(path)
    assert restored.is_direct
    assert [v.label for v in restored.volumes] == ["fuel", "surroundings"]
    np.testing.assert_allclose([v.shape.volume for v in restored.volumes],
                               [v.shape.volume for v in mesh.cad.volumes], rtol=1e-10)
    mapped = discretize(mesh)
    measured = (mapped.fractions().T @ mapped.grid.volumes).numpy()
    np.testing.assert_allclose(measured, [v.shape.volume / 1000 for v in restored.volumes], rtol=.015)
    assert any(all(s["sense"]) for s in mesh.surfaces.values())
    centers = mesh.points_cm[mesh.tetrahedra].mean(1)
    for index in np.linspace(0, len(centers) - 1, 40, dtype=int):
        owners = [i for i, v in enumerate(restored.volumes)
                  if any(s.is_inside(tuple(centers[index] * 10)) for s in v.shape.solids())]
        assert owners == [mesh.region_indices[index]]


def test_external_step_needs_no_construction_record(tmp_path):
    """Read an ordinary build123d STEP containing a curved solid."""
    solid = bd.Solid.make_torus(12, 3)
    bd.export_step(solid, tmp_path / "external.step")
    cad = CadGeometry.read_step(tmp_path / "external.step")
    assert cad.is_direct
    assert len(cad.volumes) == 1
    assert cad.volumes[0].shape.volume == pytest.approx(solid.volume, rel=1e-10)


def test_infinite_medium_on_nonconvex_mesh(curved_mesh):
    """Check the eigenvalue, power balance and a material derivative analytically."""
    geometry = discretize(curved_mesh)
    absorption = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    data = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in macro(absorption).items()}
    result = FiniteElementDiffusion(geometry.grid).solve(geometry.fractions(), data)
    assert float(result["keff"].detach()) == pytest.approx(1.25, rel=2e-10)
    derivative, = torch.autograd.grad(result["keff"], absorption)
    assert float(derivative) == pytest.approx(-.25 / .2**2, rel=2e-9)
    assert abs(float(result["neutron_balance_error"].detach())) < 1e-10
    np.testing.assert_allclose(result["power_W"].detach().numpy() / 100,
                               result["volume_cm3"].detach().numpy() / result["volume_cm3"].sum().item(), rtol=2e-9)


def test_vacuum_sphere_and_shape_derivative():
    """Compare a curved-domain eigenmode with its analytical Robin solution."""
    radius, diffusion = 4., 1 / 3
    sphere = bd.Solid.make_sphere(10 * radius)
    cad = CadGeometry.from_solids({"fuel": sphere})
    mesh = tetrahedral_mesh(cad, max_size_cm=.75, curvature_segments=32)
    scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    coordinates = torch.tensor(mesh.points_cm, dtype=torch.float64)
    geometry = discretize(mesh, coordinates=lambda: coordinates * scale)
    solver = FiniteElementDiffusion(geometry.grid)
    result = solver.solve(geometry.fractions(), macro())
    root = brentq(lambda x: x / np.tan(x) - 1 + radius / (2 * diffusion), 1e-6, np.pi - 1e-6)
    expected = .25 / (.2 + diffusion * (root / radius)**2)
    assert float(result["keff"].detach()) == pytest.approx(expected, rel=.015)
    derivative, = torch.autograd.grad(result["keff"], scale)
    values = []
    for value in (1 - 1e-4, 1 + 1e-4):
        with torch.no_grad():
            scale.fill_(value)
            values.append(float(solver.solve(geometry.fractions(), macro())["keff"]))
    assert float(derivative) == pytest.approx((values[1] - values[0]) / 2e-4, rel=2e-6)


def test_multigroup_material_calibration_on_tetrahedra():
    """Check the shared calibration interface against a two-group matrix solve."""
    solid = bd.Solid.make_box(10, 10, 10)
    cad = CadGeometry.from_solids({"material": solid},
        boundaries={"outside": {"type": "reflective", "faces": tuple(solid.faces())}})
    mapped = discretize(cad, max_size_cm=.5)
    absorption, fission = np.array([.07, .15]), np.array([.03, .1])
    scatter = np.array([[.4, .12], [.01, 1.]])
    data = {"absorption": absorption[None], "fission": fission[None],
            "nu_fission": 2.5 * fission[None], "kappa_fission": 2e8 * fission[None],
            "scatter": scatter[None], "nu_scatter": scatter[None],
            "transport": absorption[None] + scatter.sum(-1), "chi": [[1., 0.]]}
    model = CalibratedDiffusion(FiniteElementDiffusion(mapped.grid), mapped.fractions,
        data, {"capture": torch.ones((1, 1, 2), dtype=torch.float64)}, ("capture",), power_W=100.)

    def reference(correction):
        """Solve the independent homogeneous two-group balance equations.

        Parameters
        ----------
        correction : float
            Logarithmic capture multiplier.

        Returns
        -------
        float
            Dominant multiplication eigenvalue.
        """
        loss = np.diag(fission + np.exp(correction) * (absorption - fission) + scatter.sum(-1)) - scatter.T
        production = np.outer([1., 0.], 2.5 * fission)
        return np.linalg.eigvals(np.linalg.solve(loss, production)).real.max()

    result = model()
    assert float(result["keff"].detach()) == pytest.approx(reference(0.), rel=1e-9)
    assert abs(float(result["neutron_balance_error"].detach())) < 1e-9
    derivative, = torch.autograd.grad(result["keff"], model.log_factors)
    assert float(derivative) == pytest.approx((reference(1e-5) - reference(-1e-5)) / 2e-5, rel=1e-7)


def test_native_openmc_uses_analytic_cad(tmp_path):
    """Run analytic transport for a torus and sphere with an independent keff."""
    openmc.reset_auto_ids()
    groups = openmc.mgxs.EnergyGroups([0., 20e6])
    xs = openmc.XSdata("verification", groups)
    xs.order = 0
    xs.set_total([1.])
    xs.set_absorption([.2])
    xs.set_scatter_matrix(np.array([[[.8]]]))
    xs.set_fission([.1])
    xs.set_nu_fission([.25])
    xs.set_chi([1.])
    library = openmc.MGXSLibrary(groups)
    library.add_xsdata(xs)
    library.export_to_hdf5(tmp_path / "mgxs.h5")
    materials = []
    for label in ("fuel", "surroundings"):
        material = openmc.Material(name=label)
        material.set_density("macro", 1.)
        material.add_macroscopic("verification")
        materials.append(material)
    torus, sphere = bd.Solid.make_torus(12, 3), bd.Solid.make_sphere(22)
    cad = CadGeometry.from_solids({"fuel": torus, "surroundings": sphere - torus},
        boundaries={"outside": {"type": "reflective", "faces": tuple(sphere.faces())}})
    adapted = from_cad(cad, dict(zip(("fuel", "surroundings"), materials)))
    assert any(isinstance(s, openmc.ZTorus) for s in adapted.geometry.get_all_surfaces().values())
    assert isinstance(adapted.boundaries["outside"], openmc.Sphere)
    model = openmc.Model(adapted.geometry, openmc.Materials(materials))
    model.materials.cross_sections = tmp_path / "mgxs.h5"
    model.settings.energy_mode = "multi-group"
    model.settings.particles, model.settings.batches, model.settings.inactive = 1000, 25, 5
    model.settings.seed, model.settings.verbosity = 7481, 1
    model.settings.source = openmc.IndependentSource(space=openmc.stats.Point((1.2, 0., 0.)))
    tally = openmc.Tally()
    tally.filters, tally.scores = [openmc.CellFilter(adapted.cells)], ["flux"]
    model.tallies = openmc.Tallies([tally])
    statepoint = model.run(cwd=tmp_path, threads=1, output=False,
                           geometry_debug=True,
                           openmc_exec=str(Path(sys.prefix) / "bin/openmc"))
    with openmc.StatePoint(statepoint) as state:
        estimates = state.k_generation[5:]
        mean, error = estimates.mean(), estimates.std(ddof=1) / np.sqrt(len(estimates))
        assert abs(mean - 1.25) < max(6 * error, .015)
        assert (state.get_tally(id=tally.id).mean > 0).all()


@pytest.mark.parametrize("kind", ("nonconvex", "tilted_cone"))
def test_geouned_preserves_nonextruded_material_ownership(kind):
    """Compare converted native cells with actual BREP membership.

    Parameters
    ----------
    kind : str
        Independent nonconvex or curved solid requiring general CAD conversion.
    """
    openmc.reset_auto_ids()
    if kind == "nonconvex":
        core = bd.Solid.make_box(12, 5, 8).fuse(bd.Solid.make_box(5, 12, 8))
    else:
        core = bd.Solid.make_cone(9, 3, 15).rotate(bd.Axis.Y, 32)
    sphere = bd.Solid.make_sphere(25)
    cad = CadGeometry.from_solids({"core": core, "surroundings": sphere - core})
    native = from_cad(cad, {"core": None, "surroundings": None})
    assert all(s.boundary_type == "vacuum" for s in native.boundaries.values())
    rng = np.random.default_rng(8415)
    for point in rng.uniform(-2.5, 2.5, (300, 3)):
        actual = [v.label for v in cad.volumes if any(s.is_inside(tuple(point * 10)) for s in v.shape.solids())]
        converted = [cell.name for cell in native.cells if point in cell.region]
        assert actual == converted


def test_unsupported_cad_is_rejected_without_approximation(curved_mesh):
    """Reject unsupported surfaces before invoking the converter."""
    with pytest.raises(NotImplementedError, match="axis-aligned torus"):
        from_cad(curved_mesh.cad, {"fuel": None, "surroundings": None})
    loft = bd.Solid.make_loft([bd.Wire.make_circle(8),
        bd.Wire.make_ellipse(10, 4, bd.Plane(origin=(0, 0, 20)))])
    with pytest.raises(NotImplementedError, match="cannot preserve"):
        from_cad(CadGeometry.from_solids({"loft": loft}), {"loft": None})


def test_native_rejects_a_boundary_surface_shared_with_an_internal_interface():
    """Reject a boundary tag that would also kill particles at an interface."""
    openmc.reset_auto_ids()
    bottom = bd.Solid.make_box(20, 10, 10)
    top = bd.Solid.make_box(10, 10, 10, bd.Plane(origin=(0, 0, 10)))
    cad = CadGeometry.from_solids({"bottom": bottom, "top": top})
    with pytest.raises(NotImplementedError, match="exterior and internal"):
        from_cad(cad, {"bottom": None, "top": None})
