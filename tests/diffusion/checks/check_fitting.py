"""Known inverse problem and target-separability checks for the generic fitter."""
import json
from pathlib import Path
import tempfile

import torch

from diffusion_surrogate import Diffusion, Grid, Materials
from optimization.fitting import fit
from runtime.io import save_json as save
from diffusion_surrogate.solver import tensor

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


class Example(torch.nn.Module):
    def __init__(self):
        """Create two log-valued coefficients for the fitting fixture."""
        super().__init__()
        self.log_values = torch.nn.Parameter(torch.tensor([.1, -.2], dtype=torch.float64))

    def forward(self):
        """Produce dependent and excluded manufactured observables.

        Returns
        -------
        dict of str to torch.Tensor
            Sum, ratio, doubled sum and an explicitly excluded response.
        """
        a, b = self.log_values.exp().unbind()
        return {'sum': a+b, 'ratio': a/b, 'twice_sum': 2*(a+b), 'excluded': 1000*a}


reference = {'observables': {'sum': {'mean': 5., 'scale': .01}, 'ratio': {'mean': 2/3, 'scale': .01},
                             'twice_sum': {'mean': 10., 'scale': .02},
                             'excluded': {'mean': 999999., 'scale': 1.}}}
with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as directory:
    path = Path(directory)/'reference.json'
    save(path, reference)
    model = Example()
    result = fit(model, ['sum', 'ratio'], path)
    error = float((model.log_values.detach().exp()-torch.tensor([2., 3.])).abs().max())
    if error > 1e-7 or not result['locally_identifiable']:
        raise AssertionError('Generic file-backed fitting failed the known solution')
    other = Example()
    adam = torch.optim.Adam(other.parameters(), lr=.04)
    fit(other, ['sum', 'ratio'], path, optimizer=adam, steps=2000)
    adam_error = float((other.log_values.detach().exp()-torch.tensor([2., 3.])).abs().max())
    if adam_error > 1e-6:
        raise AssertionError(f'Caller-provided Torch optimizer missed the solution by {adam_error}')
    duplicate = fit(Example(), ['sum', 'twice_sum'], reference)
    if duplicate['locally_identifiable'] or duplicate['jacobian_rank'] != 1:
        raise AssertionError('Redundant targets falsely identify two parameters')
    rejected = False
    try:
        fit(Example(), ['sum'], reference)
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError('Underdetermined fitting was not rejected')
    # Correlated observations must also work through the generic covariance input.
    correlated = Example()
    cov = torch.tensor([[.0001, .00003], [.00003, .0001]], dtype=torch.float64)
    fit(correlated, ['sum', 'ratio'], reference, covariance=cov)
    covariance_error = float((correlated.log_values.detach().exp()-torch.tensor([2., 3.])).abs().max())
    if covariance_error > 1e-7:
        raise AssertionError('Correlated reference fit missed the known solution')

rates = {'absorption': .2, 'fission': .08, 'nu_fission': .2, 'kappa_fission': .08,
         'scatter_p1': 0.}
microscopic = {key: torch.full((2, 1, 1), value, dtype=torch.float64) for key, value in rates.items()}
microscopic.update({key: torch.full((2, 1, 1, 1), .5, dtype=torch.float64) for key in ('scatter', 'nu_scatter')})
materials = Materials(tensor([[1.], [1.]]), microscopic, tensor([[1.], [1.]]), shared_fuel=True)
grid = Grid((-1., -1.), (1., 1.), (4, 4), ('reflective',)*4)
capture_factor = tensor(1.3).requires_grad_()
diffusion_factor = tensor(.6).requires_grad_()
fission_factor = tensor(1.2).requires_grad_()
solved = Diffusion(grid, materials).solve(tensor([[0., 0.]]), tensor([.3]),
    capture_scale=capture_factor, diffusion_scale=diffusion_factor.expand(2, 1),
    fission_scale=fission_factor.expand(2, 1))
absorption = .08*1.2+.12*1.3
expected_k = .2*1.2/absorption
dk_capture, dk_diffusion, dk_fission = torch.autograd.grad(solved['keff'], (capture_factor, diffusion_factor, fission_factor))
capture_gradient_error = abs(float(dk_capture)-(-.2*1.2*.12/absorption**2))
fission_gradient_error = abs(float(dk_fission)-(.2*.12*1.3/absorption**2))
normalization_error = abs(float(solved['region_flux_per_source'].detach().sum())-1/absorption)
if max(abs(float(solved['keff'].detach())-expected_k), capture_gradient_error,
       fission_gradient_error, abs(float(dk_diffusion)), normalization_error,
       abs(float(solved['neutron_balance_error'].detach()))) > 1e-10:
    raise AssertionError('Corrected-material homogeneous solution or source normalization is wrong')

record = {'passed': True, 'lbfgs_parameter_error': error, 'adam_parameter_error': adam_error,
          'covariance_parameter_error': covariance_error, 'redundant_target_rank': duplicate['jacobian_rank'],
          'underdetermined_rejected': rejected, 'unselected_observation_ignored': True,
          'homogeneous_keff_error': abs(float(solved['keff'].detach())-expected_k),
          'capture_gradient_error': capture_gradient_error,
          'fission_gradient_error': fission_gradient_error,
          'reflecting_diffusion_gradient': float(dk_diffusion), 'source_normalization_error': normalization_error}
save(ROOT/'results/fitting_checks.json', record)
print(json.dumps(record, indent=2))
