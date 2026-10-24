"""Known 3D inverse problem and fitting-cost check; no Monte Carlo transport."""
import json
from pathlib import Path
import time

import torch

from diffusion_surrogate import Grid3D, Diffusion3D, Response, cylinder_regions, cartesian_responses
from optimization.fitting import fit
from diffusion_surrogate.calibration import CalibratedDiffusion
from runtime.io import save_json as save
from diffusion_surrogate.solver import tensor

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
grid = Grid3D(tuple(torch.linspace(low, high, 9, dtype=torch.float64) for low, high in
                    ((-1.4, 1.4), (-1.4, 1.4), (-6., 6.))),
              ('reflective',)*4+('vacuum',)*2)
centers = tensor([[-.65, -.65], [.65, -.65], [-.65, .65], [.65, .65]])
bounds = tensor([[-6., 6.]]*4)


def geometry():
    """Map the fixture's cylinders onto its three-dimensional grid.

    Returns
    -------
    torch.Tensor
        Cell-by-region cylinder and background fractions.
    """
    return cylinder_regions(grid, centers, tensor([.36]*4), bounds)


absorption = tensor([[.08, .012]]*4+[[.004, .0002]])
fission = tensor([[.045, .006]]*4+[[0., 0.]])
scatter = tensor([[[.3, .0001], [.06, .25]]]*4+[[[1.2, .0002], [.08, .28]]])
moment = tensor([[.05, .15]]*4+[[.4, .13]])
macro = {'absorption': absorption, 'fission': fission, 'nu_fission': 2.4*fission,
         'kappa_fission': 2e8*fission, 'scatter': scatter, 'nu_scatter': scatter,
         'scatter_p1': moment, 'transport': absorption+scatter.sum(-1)-moment,
         'chi': tensor([[0., 1.]]*5)}
names = ("fission", "capture", "diffusion")
bases = {name: torch.zeros((3, 5, 2), dtype=torch.float64) for name in (*names, "scatter")}
bases["boundary"] = torch.zeros((3, 6, 2), dtype=torch.float64)
for i, name in enumerate(names):
    bases[name][i] = 1.
model = CalibratedDiffusion(Diffusion3D(grid), geometry, macro, bases, names, power_W=100.,
    responses=lambda output: cartesian_responses(output, grid.edges,
        [[-1.4, 0., 1.4], [-1.4, 0., 1.4], [-6., -3., 0., 3., 6.]], [0., .625, 2e7], 100.))
truth = tensor([.95, 1.1, 1.2]).log()
with torch.no_grad():
    model.log_factors.copy_(truth)
    reference_output = model()
    model.log_factors.zero_()
target_names = ['keff', 'region_flux_per_source', 'capture_per_source', 'power_shape', 'flux_shape_0', 'flux_shape_1']
reference = {'observables': {}}
for name in target_names:
    response = reference_output[name]
    value = response.values if isinstance(response, Response) else response
    entry = {'mean': value.tolist(), 'scale': (.01*value.abs()).clamp_min(1e-10).tolist()}
    if isinstance(response, Response):
        entry['layout'] = response.layout
    reference['observables'][name] = entry
started = time.perf_counter()
result = fit(model, target_names, reference, steps=50, jacobian_mode='forward',
             parameter_labels={'log_factors': names})
elapsed = time.perf_counter()-started
error = float((model.log_factors.detach()-truth).abs().max())
assert error < 1e-6 and result['locally_identifiable']
record = {'passed': True, 'mesh': grid.shape, 'groups': 2, 'fit_wall_s': elapsed,
          'evaluations': result['function_evaluations'], 'free_parameters': result['free_parameters'],
          'observation_components': result['selected_components'],
          'jacobian_mode': result['jacobian_mode'], 'jacobian_rank': result['jacobian_rank'],
          'log_parameter_max_error': error, 'loss': result['loss'],
          'native_transport_histories': 0,
          'scope': 'Known inverse problem; not a fitted OpenMC prediction or equilibrium-cycle test'}
save(ROOT/'results/fitting3d_checks.json', record)
print(json.dumps(record, indent=2))
