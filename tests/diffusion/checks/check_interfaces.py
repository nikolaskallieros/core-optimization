"""Known-answer checks of the public Torch interfaces; no neutron histories."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import time

import torch

from optimization.design import Snapshot, optimize
from optimization.fitting import Target, fit
from optimization.objectives import first_crossing
from optimization.solvers import minimize

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
started = time.perf_counter()
example = runpy.run_path(str(ROOT/'fixtures/torch_interfaces.py'))
model, design, fitted, coefficients, result = example['run']()
expected = torch.tensor([[1., 2., 3.], [-1., -2., -3.]], dtype=torch.float64)
position_error = float((design['positions'].detach()-expected).abs().max())
assert position_error < 1e-6 and result['objective_after'] < 1e-12
assert fitted['optimizer_steps'] == result['optimizer_steps'] == 1
assert fitted['optimizer_iterations'] > 0 and result['optimizer_iterations'] > 0
assert len(result['gradient']['positions']) == 2 and len(result['gradient']['positions'][0]) == 3
assert all(not p.requires_grad for p in model.parameters())

# History objectives use caller-selected coordinates and thresholds, not k=1.
shift = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
coordinates = torch.tensor([0., 2., 4.], dtype=torch.float64)
crossing, _, outside = first_crossing(coordinates, 5+shift-2*coordinates, target=1.2)
assert abs(float(crossing.detach())-2.15) < 1e-12 and not outside
assert abs(float(torch.autograd.grad(crossing, shift)[0])-.5) < 1e-12

# Forward Jacobians must receive the same positional and keyword inputs.
other = example['LinearResponse']()
forward = fit(other, [Target('response', transform='identity')],
              {'observables': {'response': {'mean': [[3., 4., 5.]]*2, 'scale': .1}}},
              args=(torch.ones((2, 3), dtype=torch.float64),), kwargs={'offset': 1.},
              initial=coefficients, jacobian_mode='forward')
assert forward['initialization']['kind'] == 'selected_fit' and forward['jacobian_rank'] == 3
assert abs(forward['loss']-fitted['loss']) < 1e-8


class Reduced(torch.nn.Module):
    def __init__(self, design):
        """Bind dictionary design parameters to one fitted scalar.

        Parameters
        ----------
        design : torch.nn.Module
            Live design parameters used by the fixture.
        """
        super().__init__()
        self.design = design
        self.a = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))

    def forward(self):
        """Evaluate the fixture's quadratic loss and coefficient marker.

        Returns
        -------
        dict of str to torch.Tensor
            Loss and marker retaining design and coefficient gradients.
        """
        return {'loss': 1+self.a*(self.design['x']-1).square(), 'marker': self.a}


def parameters():
    """Build active, unused and fixed parameter fixtures.

    Returns
    -------
    torch.nn.ParameterDict
        Parameter dictionary exposing all three optimization-interface cases.
    """
    return torch.nn.ParameterDict({
        'x': torch.nn.Parameter(torch.tensor(0., dtype=torch.float64)),
        'unused': torch.nn.Parameter(torch.ones(2, dtype=torch.float64)),
        'fixed': torch.nn.Parameter(torch.tensor(7., dtype=torch.float64), requires_grad=False),
    })


def trainable(design):
    """Select fixture parameters whose gradients are enabled.

    Parameters
    ----------
    design : torch.nn.Module
        Live design parameters used by the fixture.

    Returns
    -------
    list of torch.nn.Parameter
        Trainable parameters in module order.
    """
    return [p for p in design.parameters() if p.requires_grad]


# Multiple cheap steps must not add detailed calls; test a different objective
# and caller projection on modules with frozen and unused design parameters.
calls = []


def detailed(design, previous):
    """Record a manufactured detailed-model evaluation.

    Parameters
    ----------
    design : torch.nn.Module
        Live design parameters used by the fixture.
    previous : Snapshot or None
        Prior reference accepted by the refresh interface.

    Returns
    -------
    Snapshot
        Manufactured loss/marker values and their observation scales.
    """
    output = {'loss': 1+2*(design['x']-1).square(), 'marker': torch.tensor(2., dtype=torch.float64)}
    calls.append(float(design['x']))
    return Snapshot(output, {'observables': {key: {'mean': float(value), 'scale': .1}
                                            for key, value in output.items()}})


design = parameters()
optimizer = torch.optim.SGD(design.parameters(), lr=.1)
outer = optimize(design, detailed, lambda design, _: Reduced(design), ['loss', 'marker'],
                 lambda out: out['loss'], optimizer=optimizer, steps=1, surrogate_steps=3,
                 project=lambda design, _: design['x'].clamp_(max=.5), accept=lambda *_: True)
assert len(calls) == 2 and float(design['x'].detach()) == .5
assert outer['proposals'][0]['optimization']['optimizer_steps'] == 3
assert outer['proposals'][0]['gradient']['unused'] == [0., 0.]
assert float(design['fixed']) == 7. and not design['fixed'].requires_grad
assert outer['fits'][-1]['initialization']['kind'] == 'selected_fit'
assert outer['model'].a.requires_grad and float(outer['model'].a.detach()) == 2.

# An overshooting momentum step selects an earlier iterate AND its momentum.
design = torch.nn.ParameterDict({'x': torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))})
optimizer = torch.optim.SGD(design.parameters(), lr=.3, momentum=.9)
best = minimize(design, lambda: design['x'], lambda x: (x-1).square(),
                optimizer=optimizer, steps=4, keep_best=True)
assert best['selected_step'] == 2
assert abs(float(design['x'].detach())-1.38) < 1e-12
assert abs(float(optimizer.state[design['x']]['momentum_buffer'])+2.6) < 1e-12

# Optimizer parameter groups retain their ordinary Torch learning rates.
design = torch.nn.ParameterDict({name: torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
                                 for name in ('left', 'right')})
optimizer = torch.optim.SGD([{'params': [design['left']], 'lr': .1},
                             {'params': [design['right']], 'lr': .2}])
minimize(design, lambda: design, lambda out: out['left'].square()+out['right'].square(),
         optimizer=optimizer)
assert abs(float(design['left'].detach())-.8) < 1e-12
assert abs(float(design['right'].detach())-.6) < 1e-12
optimizer = torch.optim.SGD([design['left']], lr=.1)
minimize(design, lambda: design, lambda out: out['left'].square()+out['right'].square(),
         optimizer=optimizer)
assert abs(float(design['left'].detach())-.64) < 1e-12
assert abs(float(design['right'].detach())-.6) < 1e-12 and design['right'].requires_grad

# Exceptions must not leave the design at an unverified candidate.
design = parameters()
optimizer = torch.optim.SGD(trainable(design), lr=.1, momentum=.9)
before = deepcopy(design.state_dict())
failure_calls = []


def failing_detailed(design, previous):
    """Inject a deliberate failure only after the initial snapshot.

    Parameters
    ----------
    design : torch.nn.Module
        Live design parameters used by the fixture.
    previous : Snapshot or None
        Prior reference; a non-None value triggers the intended failure.

    Returns
    -------
    Snapshot
        Initial manufactured snapshot when no prior reference exists.

    Raises
    ------
    RuntimeError
        A refresh is attempted after the first reference.
    """
    failure_calls.append(float(design['x']))
    if previous is not None:
        raise RuntimeError('intentional reference failure')
    return detailed(design, previous)


try:
    optimize(design, failing_detailed, lambda design, _: Reduced(design), ['loss', 'marker'],
             lambda out: out['loss'], optimizer=optimizer, steps=1, accept=lambda *_: True)
except RuntimeError as error:
    assert str(error) == 'intentional reference failure'
else:
    raise AssertionError('The failed detailed evaluation was hidden')
assert len(failure_calls) == 2 and not optimizer.state
assert all(torch.equal(value, design.state_dict()[name]) for name, value in before.items())
try:
    minimize(design, lambda: design['x'], lambda x: x/0., optimizer=optimizer)
except ArithmeticError:
    pass
else:
    raise AssertionError('A nonfinite objective was accepted')
assert not optimizer.state
assert all(torch.equal(value, design.state_dict()[name]) for name, value in before.items())

record = {'passed': True, 'torch': torch.__version__, 'wall_s': time.perf_counter()-started,
          'example_position_max_error': position_error, 'fit_args_kwargs_and_warm_start': True,
          'lbfgs_fit_iterations': fitted['optimizer_iterations'],
          'lbfgs_design_iterations': result['optimizer_iterations'],
          'three_surrogate_steps_use_two_detailed_calls': True,
          'frozen_and_unused_parameters_preserved': True, 'ordinary_parameter_groups_and_subsets': True,
          'history_objective_threshold_and_derivative': True,
          'best_iterate_restores_matching_optimizer': True,
          'failed_reference_and_nonfinite_objective_restore_design': True,
          'native_transport_histories': 0,
          'scope': 'Exact algebraic interface checks, not a new reactor calculation'}
(ROOT/'results/interface_checks.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
