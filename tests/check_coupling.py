"""Compare the shared depletion kernels and gradients with independent references."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmc.deplete
import scipy.sparse
import torch

from runtime.paths import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from depletion.kernels import ChainMatrix, cram
from depletion.kernels._coefficients import COEFFICIENTS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation-output/coupling_primitives.json"
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

def chain_fixture():
    """Build manufactured fission, capture and decay connectivity.

    Returns
    -------
    openmc.deplete.Chain
        Four-nuclide test chain with deliberately artificial half-life and yields.
    """
    chain = openmc.deplete.Chain()
    parent, iodine, xenon, stable = [openmc.deplete.Nuclide(n) for n in
                                    ('U235', 'I135', 'Xe135', 'Xe136')]
    parent.add_reaction('fission', None, 200.e6, 1.)
    parent.add_reaction('(n,gamma)', None, 0., 1.)
    parent.yield_data = {0.: {'I135': 1.}}
    iodine.half_life = 2.e4  # Deliberately manufactured, not an evaluated datum.
    iodine.add_decay_mode('beta-', 'Xe135', 1.)
    xenon.add_reaction('(n,gamma)', 'Xe136', 0., 1.)
    chain.nuclides = [parent, iodine, xenon, stable]
    chain.nuclide_dict = {n.name: i for i, n in enumerate(chain.nuclides)}
    chain.reactions = ['fission', '(n,gamma)']
    return chain

def kernel_checks(components):
    """Compare bundled chain/CRAM kernels with native and exact results.

    Parameters
    ----------
    components : types.SimpleNamespace
        Loaded chain-matrix, CRAM solver and rational-coefficient implementations.

    Returns
    -------
    dict
        Coefficient identity, matrix errors, inventory errors and scaled derivative errors
        for CRAM16/48.
    """
    coefficients_equal = {}
    for order in (16, 48):
        alpha, theta, alpha0 = components.coefficients[order]
        native = getattr(openmc.deplete.cram, f'Cram{order}Solver')
        coefficients_equal[str(order)] = bool(
            np.array_equal(alpha, native.alpha) and np.array_equal(theta, native.theta)
            and alpha0 == native.alpha0)
        assert coefficients_equal[str(order)]
    chain = chain_fixture()
    factory = components.ChainMatrix(chain)
    rates = torch.tensor([[2.e-7, .4e-7], [0., 0.], [0., 3.e-6], [0., 0.]], requires_grad=True)
    initial = torch.tensor([1., .01, .02, 0.], requires_grad=True)
    seconds = torch.tensor(2.e4, requires_grad=True)
    matrix = factory(rates)
    native_rates = openmc.deplete.ReactionRates(['1'], factory.rate_nuclides, factory.reactions)
    native_rates[0] = rates.detach().numpy()
    reference_matrix = chain.form_matrix(native_rates[0]).toarray()
    matrix_error = float(np.max(np.abs(matrix.detach().numpy()-reference_matrix)))
    assert matrix_error < 1.e-20
    weights = torch.tensor([.3, 1.2, -.2, .6])
    records = []
    for order in (16, 48):
        n = components.cram(matrix, initial, seconds, order)
        exact = torch.matrix_exp(matrix * seconds) @ initial
        objective = n @ weights
        ad = torch.autograd.grad(objective, (rates, initial, seconds), retain_graph=True)
        reference = torch.autograd.grad(exact @ weights, (rates, initial, seconds), retain_graph=True)
        gradient_error = max(float(((a-b)/(1+b.abs())).abs().max()) for a, b in zip(ad, reference))
        native = getattr(openmc.deplete.cram, f'CRAM{order}')(
            scipy.sparse.csc_array(reference_matrix), initial.detach().numpy(), seconds.item())
        native_error = float(np.max(np.abs(n.detach().numpy()-native)))
        exact_error = float((n-exact).detach().abs().max())
        assert max(gradient_error, native_error, exact_error) < 2.e-12
        records.append({'order': order, 'native_inventory_absolute_error': native_error,
                        'matrix_exp_inventory_absolute_error': exact_error,
                        'max_scaled_gradient_error': gradient_error})
    return {'coefficients_bitwise_equal_to_openmc_016': coefficients_equal,
            'chain_matrix_absolute_error': matrix_error, 'cases': records}

if __name__ == "__main__":
    components = SimpleNamespace(ChainMatrix=ChainMatrix, cram=cram, coefficients=COEFFICIENTS)
    result = {"status": "passed", "reused_kernels": kernel_checks(components),
              "scope": "Manufactured depletion matrices and gradients; no native reactor history."}
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Passed: CRAM16/48 inventories, chain matrices and gradients.")
