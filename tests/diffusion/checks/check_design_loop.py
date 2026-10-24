"""Manufactured callback/parameter-isolation check; no OpenMC physics claim."""
import json
from pathlib import Path
import time

import torch

from optimization.design import Snapshot, optimize
from optimization.fitting import Target
import runpy

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
objective_se = runpy.run_path(str(ROOT / "fixtures/power_objective.py"))["objective_se"]


class Design(torch.nn.Module):
    def __init__(self):
        """Create the scalar design coordinate used by the loop fixture."""
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))


class Reduced(torch.nn.Module):
    def __init__(self, design):
        """Bind a scalar design to one trainable correction coefficient.

        Parameters
        ----------
        design : torch.nn.Module
            Live design parameters used by the fixture.
        """
        super().__init__()
        self.design = design
        self.log_a = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self):
        """Evaluate the manufactured quadratic loss and correction marker.

        Returns
        -------
        dict of str to torch.Tensor
            Differentiable loss and positive calibration marker.
        """
        a = self.log_a.exp()
        return {'loss': 1+a*(self.design.x-1)**2, 'marker': a}


calls, fitted_designs = [], []


def detailed(design, previous):
    """Create the manufactured high-fidelity snapshot at the current design.

    Parameters
    ----------
    design : torch.nn.Module
        Live design parameters used by the fixture.
    previous : Snapshot or None
        Previous reference snapshot accepted by the refresh callback interface.

    Returns
    -------
    Snapshot
        Manufactured outputs, observation scales and the evaluated design coordinate.
    """
    x = design.x.detach().clone()
    a = 2+.1*x
    output = {'loss': 1+a*(x-1)**2, 'marker': a}
    calls.append(float(x))
    reference = {'observables': {key: {'mean': float(value), 'scale': .01} for key, value in output.items()}}
    return Snapshot(output, reference, metadata={'x': float(x)})


def event(kind, step, record):
    """Verify that each calibration event fits only one coefficient.

    Parameters
    ----------
    kind : str
        Loop event label.
    step : int
        Current design-loop step.
    record : mapping
        Event diagnostics containing the fit report.
    """
    if kind == 'fit':
        fitted_designs.append(float(design.x.detach()))
        assert record['report']['free_parameters'] == 1


design = Design()
optimizer = torch.optim.SGD(design.parameters(), lr=.1, momentum=.3)
started = time.perf_counter()
result = optimize(design, detailed, lambda design, _: Reduced(design),
                  [Target('loss'), Target('marker')], lambda output: output['loss'],
                  optimizer=optimizer, steps=2, project=lambda *_: None,
                  accept=lambda current, candidate, proposal: proposal['step'] == 1,
                  fit_options={'steps': 25}, callback=event)
assert len(calls) == 3
assert abs(calls[1]-.4) < 1e-8
assert abs(float(design.x.detach())-calls[1]) < 1e-12
assert fitted_designs == [calls[0], calls[1], calls[1]]
assert abs(result['proposals'][0]['gradient']['x']+4) < 1e-8
assert result['fits'][1]['initial_parameter_values'] == result['fits'][0]['parameter_values']
assert all(item['jacobian_rank'] == 1 for item in result['fits'])
assert abs(result['snapshot'].metadata['x']-calls[1]) < 1e-12
assert abs(optimizer.state[design.x]['momentum_buffer'].item()+4) < 1e-8

# Recover after the final design decision was saved but its calibration was not.
# Reuse the accepted detailed state, preserve optimizer history, and refit only.
calls_before = list(calls)
momentum_before = optimizer.state[design.x]['momentum_buffer'].clone()
recovered = optimize(design, detailed, lambda design, _: Reduced(design),
                     [Target('loss'), Target('marker')], lambda output: output['loss'],
                     optimizer=optimizer, steps=0, project=lambda *_: None,
                     accept=lambda *_: False, fit_options={'steps': 25},
                     initial_fit=result['selected_fit'], initial_snapshot=result['snapshot'])
assert calls == calls_before
assert not recovered['proposals'] and len(recovered['fits']) == 1
assert torch.equal(optimizer.state[design.x]['momentum_buffer'], momentum_before)
assert abs(float(recovered['model']()['marker'].detach())-(2+.1*calls[1])) < 1e-8

# Detailed callbacks run under no_grad, but delta-method statistics still need
# their own local gradient. Compare to the analytical derivative of the p8 norm.
q = torch.arange(1., 17., dtype=torch.float64)
covariance = .004*(torch.eye(16, dtype=q.dtype)-torch.ones((16, 16), dtype=q.dtype)/16)
joint = torch.zeros((17, 17), dtype=q.dtype)
joint[1:, 1:] = covariance
axes = [{'name': name, 'edges': edges, 'unit': 'cm'} for name, edges in
        zip(('x', 'y', 'z'), ([-1., 0., 1.], [-1., 0., 1.], [-2., -1., 0., 1., 2.]))]
reference = {'observables': {'power_shape': {'layout': {'axes': axes}}},
             'statistics': {'mean': [1.]+q.tolist(), 'covariance': joint.tolist()}}
state = Snapshot({}, {}, state={'reference': reference})
p = q/q.sum()
moment = p.pow(8).sum()
gradient = 16**(7/8)*(p.pow(7)-moment)/(moment**(7/8)*q.sum())
expected_se = float((gradient@covariance@gradient).sqrt())
with torch.no_grad():
    measured_se = objective_se(state)
assert abs(measured_se-expected_se) < 1e-14
record = {'passed': True, 'wall_s': time.perf_counter()-started, 'detailed_designs': calls,
          'fitted_designs': fitted_designs, 'retained_design': float(design.x.detach()),
          'warm_start_verified': True, 'rejected_step_and_optimizer_restored': True,
          'final_fit_recovery_without_new_detailed_solve': True,
          'objective_se_under_no_grad': measured_se, 'analytical_objective_se': expected_se,
          'physics': 'Manufactured verification of callbacks, gradients, warm starts and rollback only'}
(ROOT/'results/design_loop_checks.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
