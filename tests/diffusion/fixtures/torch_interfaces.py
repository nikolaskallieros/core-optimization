"""Executable interface example with exact algebraic answers; no transport.

From the project root: .venv/bin/python tests/diffusion/fixtures/torch_interfaces.py
Replace LinearResponse with a Torch diffusion model without changing the APIs.
"""
import torch

from optimization.fitting import Target, fit
from optimization.solvers import minimize
from optimization.fitting import selected_fit


class LinearResponse(torch.nn.Module):
    def __init__(self):
        """Create three trainable log-scale coefficients."""
        super().__init__()
        self.log_scale = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))

    def forward(self, positions, *, offset=0.):
        """Apply fitted positive component scales and an additive offset.

        Parameters
        ----------
        positions : torch.Tensor
            Input values with three components on the trailing axis.
        offset : float or torch.Tensor, optional
            Additive response offset.

        Returns
        -------
        dict of str to torch.Tensor
            Scaled response retaining input and coefficient derivatives.
        """
        return {'response': positions*self.log_scale.exp()+offset}


def run():
    """Fit ordinary Torch parameters and optimize a separate live design.

    Returns
    -------
    tuple
        Fitted model, design, fit result, selected coefficients and optimization result.
    """
    model = LinearResponse()
    inputs = torch.ones((2, 3), dtype=torch.float64)
    reference = {'observables': {'response': {'mean': [[3., 4., 5.]]*2, 'scale': .1}}}
    fitted = fit(model, [Target('response', transform='identity')], reference,
                 args=(inputs,), kwargs={'offset': 1.})
    coefficients = selected_fit(fitted)  # fit(next_model, ..., initial=coefficients)

    design = torch.nn.ParameterDict({
        'positions': torch.nn.Parameter(torch.zeros((2, 3), dtype=torch.float64)),
    })
    target = torch.tensor([[2., 6., 12.], [-2., -6., -12.]], dtype=torch.float64)
    optimizer = torch.optim.LBFGS(design.parameters(), max_iter=25,
                                  line_search_fn='strong_wolfe', tolerance_grad=1e-10,
                                  tolerance_change=1e-14)
    # The callable closes over a fitted model: freeze those coefficients once.
    # Only the design inputs are optimized. Both are ordinary Torch modules.
    model.requires_grad_(False)
    result = minimize(design, lambda: model(design['positions']),
                      lambda output: (output['response']-target).square().sum(),
                      optimizer=optimizer, steps=1)
    return model, design, fitted, coefficients, result


if __name__ == '__main__':
    torch.set_num_threads(1)
    model, design, fitted, coefficients, result = run()
    print('Fitted scales:', model.log_scale.exp().detach())
    print('Selected positions:', design['positions'].detach())
    print('Objective:', result['objective_after'])
    print('Optimizer iterations:', result['optimizer_iterations'])
    print('Fit / search seconds:', fitted['wall_s'], result['wall_s'])
