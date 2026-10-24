"""Analytical axial leakage, material derivatives, geometry AD and 3D timings."""
import json
import math
from pathlib import Path
import time

import torch

from diffusion_surrogate import Diffusion3D, Grid3D, cylinder_regions
from runtime.io import save_json as save
from diffusion_surrogate.solver import tensor

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def one_group(absorption, production, diffusion):
    """Build one-group macroscopic data for analytic solver checks.

    Parameters
    ----------
    absorption : array_like
        Regional absorption cross sections in cm⁻¹.
    production : array_like
        Regional neutron-production cross sections in cm⁻¹.
    diffusion : array_like
        Regional diffusion coefficients in cm.

    Returns
    -------
    dict of str to torch.Tensor
        One-group macroscopic data in the solver's region/group layout.
    """
    a, f, d = (tensor(value).reshape(-1, 1) for value in (absorption, production, diffusion))
    scattering = torch.full_like(a, .3)
    return {'absorption': a, 'fission': f/2.4, 'nu_fission': f, 'kappa_fission': f*8e7,
            'scatter': scattering[..., None], 'nu_scatter': scattering[..., None],
            'transport': 1/(3*d), 'chi': torch.ones_like(a)}


grid = Grid3D(([-1., -.3, 1.], [-2., .4, 2.], [-3., -.2, 3.]), ('reflective',)*6)
absorption = tensor(.04).requires_grad_()
production = tensor(.06).requires_grad_()
solved = Diffusion3D(grid).solve(torch.ones(grid.size, 1, dtype=torch.float64),
                               one_group(absorption, production, 1.2))
dk_da, dk_df = torch.autograd.grad(solved['keff'], (absorption, production))
reflecting_errors = {'keff': abs(float(solved['keff'].detach())-1.5),
                     'absorption_gradient': abs(float(dk_da)+37.5),
                     'production_gradient': abs(float(dk_df)-25.),
                     'balance': abs(float(solved['neutron_balance_error'].detach()))}
assert max(reflecting_errors.values()) < 1e-10

# Uniform x/y and vacuum z: continuum Robin boundary D dphi/dn = -phi/2.
height, diffusion = 12., 1.2
low, high = 0., math.pi/height
for _ in range(80):
    wave = (low+high)/2
    if wave*math.tan(wave*height/2) < 1/(2*diffusion):
        low = wave
    else:
        high = wave
wave = (low+high)/2
exact_k = .06/(.04+diffusion*wave**2)
axial_checks = []
for nz in (12, 24):
    axial = Grid3D(([-1., 0., 1.], [-1., 0., 1.], torch.linspace(-height/2, height/2, nz+1)),
                   ('reflective',)*4+('vacuum',)*2)
    out = Diffusion3D(axial).solve(torch.ones(axial.size, 1, dtype=torch.float64), one_group(.04, .06, diffusion))
    average = (torch.sin(wave*axial.edges[2][1:])-torch.sin(wave*axial.edges[2][:-1]))/(wave*axial.edges[2].diff())
    observed = out['power_density_W_cm3'].detach()[0, 0]
    axial_checks.append({'z_cells': nz, 'keff': float(out['keff'].detach()),
                         'keff_relative_error': abs(float(out['keff'].detach())/exact_k-1),
                         'axial_shape_max_relative_error': float((observed/observed.mean()-average/average.mean()).abs().max())})
assert axial_checks[1]['keff_relative_error'] < axial_checks[0]['keff_relative_error']/3
assert axial_checks[1]['axial_shape_max_relative_error'] < .001

# Four pins, finite heights and genuinely unequal axial material regions.
grid = Grid3D((torch.linspace(-1.4, 1.4, 9), torch.linspace(-1.4, 1.4, 9), torch.linspace(-6., 6., 9)),
              ('reflective',)*4+('vacuum',)*2)
centers = tensor([[-.65, -.65], [.65, -.65], [-.65, .65], [.65, .65]]).repeat_interleave(2, 0)
bounds = tensor([[-5., .4], [.4, 5.]]).repeat(4, 1)
macro = one_group([.13, .14]*4+[.01], [.22, .20]*4+[0.], [.3, .32]*4+[1.2])
model = Diffusion3D(grid)


def evaluate(radius):
    """Solve the fixture while varying its shared cylinder radius.

    Parameters
    ----------
    radius : torch.Tensor
        Scalar live radius in cm, shared by eight cylinders.

    Returns
    -------
    dict
        Diffusion response fields and solver diagnostics.
    """
    fractions = cylinder_regions(grid, centers, radius.expand(8), bounds)
    return model.solve(fractions, macro)


radius = tensor(.36).requires_grad_()
start = time.perf_counter()
out = evaluate(radius)
forward_s = time.perf_counter()-start
start = time.perf_counter()
gradient = torch.autograd.grad(out['keff']+.001*out['power_W'][0], radius)[0]
backward_s = time.perf_counter()-start
step = 1e-5
with torch.no_grad():
    minus, plus = evaluate(radius-step), evaluate(radius+step)
fd = (plus['keff']+.001*plus['power_W'][0]-minus['keff']-.001*minus['power_W'][0])/(2*step)
gradient_error = abs(float(gradient-fd))/max(1., abs(float(fd)))
assert gradient_error < 1e-6
volumes = out['volume_cm3'].detach()
exact_pin_volumes = math.pi*.36**2*(bounds[:, 1]-bounds[:, 0])
assert float((volumes[:8]-exact_pin_volumes).abs().max()) < 1e-12
assert abs(float(out['neutron_balance_error'].detach())) < 1e-10
assert abs(float((out['power_density_W_cm3'].detach().flatten()*grid.volumes).sum())-100.) < 1e-10
record = {'passed': True, 'reflecting_analytic_errors': reflecting_errors,
          'vacuum_axial_analytic_keff': exact_k, 'axial_mesh_checks': axial_checks,
          'geometry_gradient': float(gradient), 'deterministic_fd': float(fd),
          'geometry_gradient_scaled_error': gradient_error, 'mesh': grid.shape,
          'regions': 9, 'forward_s': forward_s, 'backward_s': backward_s,
          'eigen_iterations': out['iterations'], 'eigen_residual': out['eigen_residual'],
          'native_transport_histories': 0, 'scope': '3D diffusion numerical verification; first derivatives only'}
save(ROOT/'results/solver3d_checks.json', record)
print(json.dumps(record, indent=2))
