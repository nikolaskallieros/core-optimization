"""Analytical geometry, material, eigenvalue and vacuum-leakage checks; no MC."""
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq
import torch

from diffusion_surrogate import Diffusion, Grid, Materials
from diffusion_surrogate.solver import tensor

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def homogeneous(groups=2, regions=3):
    """Build repeated one- or two-group manufactured material data.

    Parameters
    ----------
    groups : {1, 2}, optional
        Number of energy groups in the analytic fixture.
    regions : int, optional
        Number of regions receiving the same microscopic data.

    Returns
    -------
    Materials
        Homogeneous material model for infinite-medium and slab checks.
    """
    if groups == 2:
        absorption, fission, nu_fission, moment = [.2, .1], [.1, .02], [.3, .05], [.2, .1]
        scatter, chi = [[1., 0.], [.4, .5]], [0., 1.]
    else:
        absorption, fission, nu_fission, moment = [.2], [.1], [.4], [.1]
        scatter, chi = [[1.]], [1.]
    micro = {k: tensor(v)[None, None].expand(regions, 1, *np.shape(v)).clone()
             for k, v in dict(absorption=absorption, fission=fission, nu_fission=nu_fission,
                              kappa_fission=fission, scatter=scatter, nu_scatter=scatter, scatter_p1=moment).items()}
    return Materials(torch.ones((regions, 1), dtype=torch.float64), micro, tensor(chi)[None].expand(regions, groups).clone())


def main():
    """Check analytic diffusion values, geometric derivatives and leakage."""
    started = time.perf_counter()
    grid = Grid((-1., -1.), (1., 1.), (14, 12), ('reflective',)*4)
    centers = tensor([[-.4, -.1], [.4, .2]]).requires_grad_()
    radii = tensor([.25, .3]).requires_grad_()
    model = Diffusion(grid, homogeneous())
    answer = model.solve(centers, radii)
    exact_k = (.05+.3*.4/.2)/(.1+.4)
    exact_power = 100*radii*radii/(radii*radii).sum()
    dk = torch.autograd.grad(answer['keff'], (centers, radii), retain_graph=True)
    dp = torch.autograd.grad(answer['power_W'][0], (centers, radii), retain_graph=True)
    reference_dp = torch.autograd.grad(exact_power[0], radii)[0]
    volume = answer['volume_cm3'][:2]
    area_error = float((volume-math.pi*radii*radii).detach().abs().max())
    area_grad = torch.autograd.grad(volume.sum(), radii)[0]
    checks = {
        'circle_area_absolute_error_cm2': area_error,
        'circle_area_radius_gradient_error_cm': float((area_grad-2*math.pi*radii).detach().abs().max()),
        'homogeneous_two_group_keff_error': abs(float(answer['keff'].detach())-exact_k),
        'homogeneous_power_error_W': float((answer['power_W']-exact_power).detach().abs().max()),
        'homogeneous_geometry_keff_gradient_error': max(float(x.abs().max()) for x in dk),
        'homogeneous_translation_power_gradient_error': float(dp[0].abs().max()),
        'homogeneous_radius_power_gradient_error': float((dp[1]-reference_dp).abs().max()),
        'eigen_residual': float(answer['eigen_residual'].detach()),
    }
    # One fuel input must drive both pins, including its reverse derivative.
    data = homogeneous()
    shared = Materials(data.atom_density, data.microscopic, data.chi, shared_fuel=True)
    correction, density = tensor([.12]).requires_grad_(), tensor([1.1, .9]).requires_grad_()
    production = shared.macroscopic(correction, density)['nu_fission'][:-1].sum()
    gradients = torch.autograd.grad(production, (correction, density))
    exact_production = 2*1.1*math.exp(.12)*(.3+.05)
    checks['shared_fuel_binding_gradient_error'] = max(
        abs(float(gradients[0][0])-exact_production),
        abs(float(gradients[1][0])-exact_production/1.1), abs(float(gradients[1][1])))
    for coefficients, scales in ((tensor([0., 0.]), None), (tensor([0.]), tensor([1., 1., 1.]))):
        try:
            shared.macroscopic(coefficients, scales)
        except ValueError:
            pass
        else:
            raise AssertionError('Independent pin inputs bypassed the shared-fuel constraint')
    different = data.atom_density.clone()
    different[1] *= 1.01
    try:
        Materials(different, data.microscopic, data.chi, shared_fuel=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Unequal physical fuel inputs were silently tied')
    assert max(checks.values()) < 2e-7, checks
    # An independent slab eigenvalue checks the Marshak boundary and leakage.
    diffusion = 1/(3*(.2+1-.1))
    wave = brentq(lambda b: diffusion*b*math.tan(b)-.5, 1e-9, math.pi/2-1e-9)
    exact_vacuum = .4/(.2+diffusion*wave*wave)
    errors = []
    for nx in (16, 32):
        slab = Diffusion(Grid((-1., -1.), (1., 1.), (nx, 2), ('vacuum', 'vacuum', 'reflective', 'reflective')), homogeneous(1))
        result = slab.solve(centers.detach(), radii.detach())
        errors.append(abs(float(result['keff'])-exact_vacuum))
    assert errors[1] < errors[0]/3.5 and errors[1]/exact_vacuum < 8e-4, errors
    checks['vacuum_slab_errors_16_32'] = errors
    checks['vacuum_slab_exact_keff'] = exact_vacuum
    checks['status'] = 'passed'
    checks['wall_s'] = time.perf_counter()-started
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/solver_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
