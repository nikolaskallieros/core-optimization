"""Physical weighting and invariance checks for the equilibrium stopping rule."""
import json
from pathlib import Path

import numpy as np

from depletion.integrators import reaction_weighted_change
from runtime.io import save_json as save

ROOT = Path(__file__).resolve().parents[1]
before = np.array([[1e22, 1e13], [1e22, 1e13]])
rates = np.array([[1e-7, 1e-12], [1e-7, 1e-12]])
volumes = np.array([1., 3.])
trace = before.copy()
trace[:, 1] *= 2
trace_result = reaction_weighted_change(before, trace, rates, volumes)
assert trace_result['maximum_region'] < 1e-12
important = before.copy()
important[0, 0] *= 1.01
result = reaction_weighted_change(before, important, rates, volumes)
assert abs(result['maximum_region']-1/101) < 1e-12
assert abs(result['global']-1/401) < 1e-12
relabelled = reaction_weighted_change(before[:, ::-1], important[:, ::-1], rates[:, ::-1], volumes)
rescaled = reaction_weighted_change(before*1e-24, important*1e-24, rates*1e24, volumes*1e-6)
for name in ('global', 'maximum_region'):
    assert abs(result[name]-relabelled[name]) < 1e-15
    assert abs(result[name]-rescaled[name]) < 1e-15
record = {'passed': True, 'doubling_a_trace_inventory': trace_result,
          'one_percent_change_in_a_dominant_reacting_inventory': result,
          'nuclide_order_and_units_invariant': True,
          'scope': 'Stopping-rule algebra and physical weighting, not evidence of actual cycle convergence'}
save(ROOT/'results/equilibrium_convergence_checks.json', record)
print(json.dumps(record, indent=2))
