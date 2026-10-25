import torch

from diffusion_surrogate import Diffusion3D, Grid3D, SparseDiffusion3D
from diffusion_surrogate.solver import tensor


def test_implicit_gradient_keeps_the_same_forward_solution():
    """Verify implicit gradient keeps the same forward solution."""
    grid = Grid3D(([-1.2, -.3, .5, 1.4], [-1., -.2, 1.3], [-3., -1., 0., 3.]),
                  ("vacuum", "reflective", "vacuum", "vacuum", "vacuum", "vacuum"))
    absorption = tensor([[.04, .09], [.008, .02]])
    fission = tensor([[.012, .034], [0., 0.]])
    scatter = tensor([[[.3, .025], [.001, .5]], [[.12, .014], [.0015, .4]]])
    macro = {"absorption": absorption, "fission": fission, "nu_fission": 2.4 * fission,
             "kappa_fission": 2e8 * fission, "scatter": scatter, "nu_scatter": scatter,
             "transport": absorption + scatter.sum(-1), "chi": tensor([[.01, .99], [.01, .99]])}
    coordinate = torch.linspace(-1., 1., grid.size, dtype=torch.float64)
    sparse = SparseDiffusion3D(grid)

    def evaluate(solver, x):
        """Solve the same smooth material mixture with a selected solver.

        Parameters
        ----------
        solver : Diffusion3D
            Explicit or implicit-gradient solver under comparison.
        x : torch.Tensor
            Scalar shift controlling the manufactured material fraction.

        Returns
        -------
        dict
            Power-normalized diffusion solution and diagnostics.
        """
        fraction = torch.sigmoid(coordinate + x)
        return solver.solve(torch.stack((fraction, 1 - fraction), 1), macro, power_W=100.)

    x = tensor(.2).requires_grad_()
    with torch.no_grad():
        forward = evaluate(sparse, x)
    differentiated = evaluate(sparse, x)
    keys = ("keff", "power_W", "region_flux_per_source", "leakage_per_source")
    for key in keys:
        assert torch.equal(forward[key], differentiated[key]), key
    loss = differentiated["keff"] + .01 * differentiated["power_W"][0]
    gradient = torch.autograd.grad(loss, x)[0]
    dense = evaluate(Diffusion3D(grid), x)
    reference_gradient = torch.autograd.grad(dense["keff"] + .01 * dense["power_W"][0], x)[0]
    torch.testing.assert_close(gradient, reference_gradient, rtol=1e-8, atol=1e-10)
    step = 1e-4
    with torch.no_grad():
        low, high = evaluate(sparse, x - step), evaluate(sparse, x + step)
        finite_difference = (high["keff"] - low["keff"]
                             + .01 * (high["power_W"][0] - low["power_W"][0])) / (2 * step)
    torch.testing.assert_close(gradient, finite_difference, rtol=1e-6, atol=1e-9)
