"""Control-branch derivatives; analytic numerical fixtures, not WR-1 physics."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion_surrogate.regional import RegionalSurrogate as WR1Surrogate
from coupling.response import ResponsePoint as WR1PointModel


class LinearPoint(WR1PointModel):
    def __init__(self):
        """Create a manufactured linear boron/inventory response."""
        self.reference_boron = 1.
        self.model = torch.nn.Module()
        self.model.design = torch.nn.ParameterDict({'shift': torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))})

    def evaluate(self, inventory, boron):
        """Evaluate the analytic control fixture with one live design shift.

        Parameters
        ----------
        inventory : torch.Tensor
            Manufactured zone-by-nuclide atom densities in atoms/cm³.
        boron : torch.Tensor
            Fixture boron concentration.

        Returns
        -------
        dict of str to torch.Tensor
            Linear keff response and the supplied boron concentration.
        """
        return {"keff": .95 + .1 * inventory.sum() - .02 * boron + .04 * self.model.design['shift'],
                "boron_ppm": boron}


def test_subcritical_bound_keeps_design_derivative_and_cannot_claim_a_root():
    """Verify subcritical bound keeps design derivative and cannot claim a root."""
    inventory = torch.tensor([.2], dtype=torch.float64, requires_grad=True)
    point = LinearPoint()
    result = point.operate(inventory, allow_subcritical_trials=True)
    assert result['control_status'] == 'subcritical_trial'
    assert result['boron_ppm'] == 0 and result['critical_root'] is None
    assert result['keff'].item() == pytest.approx(.97)
    assert result['criticality_deficit'].item() == pytest.approx(.03)
    assert torch.autograd.grad(result['criticality_deficit'], inventory)[0].item() == pytest.approx(-.1)
    with pytest.raises(RuntimeError, match='No bracketed root'):
        point.operate(inventory)


def test_feasible_branch_retains_implicit_boron_derivative():
    """Verify feasible branch retains implicit boron derivative."""
    inventory = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
    result = LinearPoint().operate(inventory, allow_subcritical_trials=True)
    assert result['control_status'] == 'critical'
    assert result['boron_ppm'].item() == pytest.approx(2.5)
    assert result['keff'].item() == pytest.approx(1.)
    assert torch.autograd.grad(result['keff'], inventory, retain_graph=True)[0].item() == pytest.approx(0., abs=1e-15)
    assert torch.autograd.grad(result['boron_ppm'], inventory)[0].item() == pytest.approx(5.)


@pytest.mark.parametrize('method', ['operate', 'checkpoint_operate'])
def test_bol_must_be_critical_even_when_eol_trials_are_allowed(method):
    """Verify BOL must be critical even when EOL trials are allowed.

    Parameters
    ----------
    method : str
        Direct or checkpointed response-operation method under test.
    """
    point = LinearPoint()
    inventory = torch.tensor([.2], dtype=torch.float64, requires_grad=True)
    with pytest.raises(RuntimeError, match='No bracketed root'):
        getattr(point, method)(inventory, allow_subcritical_trials=True, beginning_of_cycle=True)
    output = getattr(point, method)(inventory, allow_subcritical_trials=True, beginning_of_cycle=False)
    assert output['control_status'] == 'subcritical_trial'
    assert output['keff'].item() == pytest.approx(.97)


def test_zero_boron_reference_needs_no_fabricated_boron_scattering():
    """Verify zero boron reference needs no fabricated boron scattering."""
    model = SimpleNamespace(case={'boron_mg_B10_per_kg': 0., 'regions': [0, 1, 2]},
                            reference={'boron_response': {
                                'atom_density': [[0., 0.]], 'region_indices': [1],
                                'microscopic': {'absorption': np.ones((1, 2, 2))},
                                'analog_sampled': [[False, False]]}})
    result = WR1Surrogate.boron_material_delta(model, torch.tensor(0., dtype=torch.float64))
    assert torch.count_nonzero(result['absorption']) == 0
    for boron in (torch.tensor(.1, dtype=torch.float64), torch.tensor(0., dtype=torch.float64, requires_grad=True)):
        with pytest.raises(ValueError, match='positive-boron calibration'):
            WR1Surrogate.boron_material_delta(model, boron)


@pytest.mark.parametrize('amount', [.2, 1.])
def test_recomputed_control_matches_direct_inventory_and_design_gradients(amount):
    """Verify recomputed control matches direct inventory and design gradients.

    Parameters
    ----------
    amount : float
        Synthetic initial inventory value for comparing recomputed and direct derivatives.
    """
    point = LinearPoint()
    inventory = torch.tensor([amount], dtype=torch.float64, requires_grad=True)
    results = []
    for method in (point.operate, point.checkpoint_operate):
        result = method(inventory, allow_subcritical_trials=True)
        probe = result['keff'].square() + .03 * result['boron_ppm'] + result['criticality_deficit'].square()
        results.append((probe.detach(), torch.autograd.grad(probe, (inventory, point.model.design['shift']))))
    torch.testing.assert_close(results[0][0], results[1][0])
    for direct, replayed in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(direct, replayed)
