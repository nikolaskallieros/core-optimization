"""Torch translation of OpenMC 0.15.3 IPFCramSolver.__call__.

Source: https://github.com/openmc-dev/openmc/blob/27e38e894697bb32a1dac7848d2618818b6b8daf/openmc/deplete/cram.py at
27e38e894697bb32a1dac7848d2618818b6b8daf (MIT; LICENSE in this directory).
The pole order, IPF recurrence and final scaling are unchanged. Dense batched
solves replace SciPy sparse solves for small chains. Real autograd through
complex128 solves supplies the conjugate-transpose VJP; no matrix is detached.
This implementation is not a sparse solver for production-scale chains.
"""
import torch
from ._coefficients import COEFFICIENTS


def cram(matrix, initial, seconds, order=48):
    """Advance a frozen depletion operator with the IPF CRAM recurrence.

    Parameters
    ----------
    matrix : torch.Tensor
        CPU float64 depletion matrix in s⁻¹ with optional leading batch axes.
    initial : torch.Tensor
        CPU float64 initial inventory with matching batch and nuclide axes.
    seconds : float or torch.Tensor
        Elapsed seconds, scalar or broadcastable over batch axes.
    order : {16, 48}, optional
        Order of the bundled CRAM rational approximation.

    Returns
    -------
    torch.Tensor
        Unclipped final inventory in the same unit as the input, retaining matrix, inventory
        and duration derivatives.

    Raises
    ------
    TypeError
        Matrix or inventory precision is not float64 Torch.
    ValueError
        Device, axes or finite-value checks fail.
    KeyError
        The requested approximation order is not bundled.

    Notes
    -----
    Matrix, inventory and elapsed time remain differentiable. Inputs use CPU float64; output
    inventories are not clipped, matching the native routine. Negative time is allowed
    mathematically for derivative checks only.
    """
    for name, value in (('matrix', matrix), ('initial', initial)):
        if value.dtype != torch.float64:
            raise TypeError(name + ' must be a float64 tensor')
        if value.device.type != 'cpu':
            raise ValueError('Only CPU depletion is implemented')
    if matrix.ndim < 2 or matrix.shape[-2:] != (initial.shape[-1],) * 2:
        raise ValueError('Matrix and inventory sizes differ')
    dt = torch.as_tensor(seconds, dtype=torch.float64, device='cpu')
    if not torch.isfinite(dt).all() or not torch.isfinite(matrix).all() or not torch.isfinite(initial).all():
        raise ValueError('Nonfinite depletion input')
    alpha_values, theta_values, alpha0 = COEFFICIENTS[order]
    alpha = torch.tensor(alpha_values, dtype=torch.complex128)
    theta = torch.tensor(theta_values, dtype=torch.complex128)
    scaled = (matrix * dt[..., None, None]).to(torch.complex128)
    identity = torch.eye(matrix.shape[-1], dtype=torch.complex128)
    y = initial
    for residue, pole in zip(alpha, theta):
        solution = torch.linalg.solve(scaled - pole * identity, y.to(torch.complex128).unsqueeze(-1))
        y = y + 2 * (residue * solution.squeeze(-1)).real
    return y * alpha0
