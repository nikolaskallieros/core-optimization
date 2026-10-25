"""Independent material-calibration checks across solver and geometry choices."""
import math

import pytest
import torch

from diffusion_surrogate import CalibratedDiffusion, Diffusion3D, Grid3D, SparseDiffusion3D
from diffusion_surrogate.hexagonal import C6Grid
from diffusion_surrogate.solver import tensor
from optimization.fitting import fit


def material_tables(*, moderator=False):
    """Supply manufactured one-group data with optional nonfissile material.

    Parameters
    ----------
    moderator : bool, optional
        Append a second material with capture and scattering but no fission.

    Returns
    -------
    dict of str to torch.Tensor
        Region/group macroscopic data in cm⁻¹, with kappa_fission in eV/cm.
    """
    absorption = tensor([[.08], [.01]] if moderator else [[.08]])
    fission = tensor([[.06], [0.]] if moderator else [[.06]])
    scatter = tensor([[[.2]]] * len(absorption))
    return {"absorption": absorption, "fission": fission, "nu_fission": 2.5 * fission,
            "kappa_fission": 2e8 * fission, "scatter": scatter, "nu_scatter": scatter,
            "transport": absorption + scatter.sum(-1), "chi": torch.ones_like(fission)}


@pytest.mark.parametrize("backend", ["cartesian_dense", "cartesian_sparse", "triangular_sparse"])
def test_same_calibration_recovers_analytical_material_factor_on_each_mesh(backend):
    """Recover one known fission correction from an infinite-medium eigenvalue.

    Parameters
    ----------
    backend : str
        Cartesian dense/sparse or triangular-prism sparse solver selection.
    """
    if backend == "triangular_sparse":
        grid = C6Grid(2., 2, [0., 1., 3.], radial_boundary="reflective",
                      axial_boundaries=("reflective", "reflective"))
    else:
        grid = Grid3D(([0., .5, 2.], [0., 1., 3.], [0., 1., 3.]), ("reflective",) * 6)
    solver = Diffusion3D(grid) if backend == "cartesian_dense" else SparseDiffusion3D(grid)
    geometry = lambda: torch.ones((grid.size, 1), dtype=torch.float64)
    macro = material_tables()
    model = CalibratedDiffusion(solver, geometry, macro, {"fission": tensor([[[1.]]])},
                                ["fission"], power_W=100.)
    factor = 1.2
    expected = 2.5 * .06 * factor / (.02 + .06 * factor)
    result = fit(model, ["keff"], {"observables": {"keff": {"mean": expected, "scale": .001}}}, steps=20)
    assert result["free_parameters"] == 1
    assert float(model.log_factors.detach()[0]) == pytest.approx(math.log(factor), abs=1e-8)
    output = model()
    assert float(output["keff"].detach()) == pytest.approx(expected, abs=1e-9)
    assert float(output["leakage_per_source"].abs().max().detach()) < 1e-12
    assert model.macroscopic()[1] == 1.
    torch.testing.assert_close(macro["fission"], tensor([[.06]]))


class MaterialMixture(torch.nn.Module):
    """A live geometry parameter owned by the design, outside calibration."""

    def __init__(self):
        """Initialize a differentiable material fraction coordinate."""
        super().__init__()
        self.coordinate = torch.nn.Parameter(tensor(.3))

    def forward(self):
        """Return complementary fuel and moderator fractions in one cell.

        Returns
        -------
        torch.Tensor
            One cell and two material fractions with a live geometry derivative.
        """
        fuel = torch.sigmoid(self.coordinate)
        return torch.stack((fuel, 1 - fuel)).reshape(1, 2)


def test_calibration_does_not_optimize_geometry_but_retains_design_derivatives():
    """Fit only material coefficients while preserving the live geometry graph."""
    grid = Grid3D(([0., 1.], [0., 1.], [0., 1.]), ("reflective",) * 6)
    geometry = MaterialMixture()
    model = CalibratedDiffusion(Diffusion3D(grid), geometry, material_tables(moderator=True),
                                {"fission": torch.ones((1, 2, 1), dtype=torch.float64)},
                                ["fission"], power_W=100.)
    initial = geometry.coordinate.detach().clone()
    assert list(dict(model.named_parameters())) == ["log_factors"]
    with torch.no_grad():
        model.log_factors.fill_(math.log(1.1))
        target = float(model()["keff"])
        model.log_factors.zero_()
    fit(model, ["keff"], {"observables": {"keff": {"mean": target, "scale": .001}}}, steps=20)
    torch.testing.assert_close(geometry.coordinate, initial, rtol=0., atol=0.)
    assert float(model.log_factors.detach()[0]) == pytest.approx(math.log(1.1), abs=1e-8)
    derivative, = torch.autograd.grad(model()["keff"], geometry.coordinate)
    assert float(derivative) > 0.


def test_unknown_material_correction_cannot_silently_enter_a_fit():
    """Reject a misspelled correction instead of fitting an inactive parameter."""
    grid = Grid3D(([0., 1.], [0., 1.], [0., 1.]))
    with pytest.raises(ValueError, match="Unknown correction bases"):
        CalibratedDiffusion(Diffusion3D(grid), lambda: tensor([[1.]]), material_tables(),
                            {"fison": tensor([[[1.]]])}, ["fission"], power_W=1.)
