"""Sparse first derivatives of the original OpenMC chain and IPF CRAM48.

Chain coefficients and rational-approximation constants come from OpenMC.
Backward reconstructs matrices from compact rate/state checkpoints and keeps
only one zone's poles at a time. This implements first derivatives, not double backward.
"""
import numpy as np
import torch
from scipy.sparse import csc_array, eye_array
from scipy.sparse.linalg import splu
from torch.autograd.function import once_differentiable

import openmc.deplete
from openmc.deplete.cram import CRAM48, Cram48Solver

from depletion.integrators import CECMDepleter, DepletionSettings


class SparseChain(torch.nn.Module):
    """Compile the linear rate-to-matrix map using OpenMC's own builder."""
    def __init__(self, chain, rate_nuclides, reactions):
        """Compile sparse rate-to-matrix coefficients using OpenMC's chain builder.

        Parameters
        ----------
        chain : openmc.deplete.Chain
            Evaluated chain supplying decay constants, branching, reaction connectivity and
            fission yields.
        rate_nuclides : sequence of str
            Parent nuclide ordering in supplied reaction-rate arrays.
        reactions : sequence of str
            Reaction ordering in supplied rate arrays.
        """
        super().__init__()
        self.names = tuple(n.name for n in chain.nuclides)
        self.rate_nuclides, self.reactions = tuple(rate_nuclides), tuple(reactions)
        self.size = len(self.names)
        decay = chain.decay_matrix.tocoo()
        fission_yields = chain.get_default_fission_yields()
        reaction_rows, reaction_cols, weights, parameters = [], [], [], []
        for parent, name in enumerate(self.rate_nuclides):
            rate = openmc.deplete.ReactionRates(["0"], [name], self.reactions)[0]
            nuclide = chain.nuclides[chain.nuclide_dict[name]]
            for reaction in sorted({r.type for r in nuclide.reactions}):
                index = self.reactions.index(reaction)
                rate[0, index] = 1.
                coefficient = chain.form_rxn_matrix(rate, fission_yields).tocoo()
                rate[0, index] = 0.
                reaction_rows.extend(coefficient.row)
                reaction_cols.extend(coefficient.col)
                weights.extend(coefficient.data)
                parameters.extend([parent * len(self.reactions) + index] * coefficient.nnz)
        linear = np.concatenate((decay.row * self.size + decay.col,
                                 np.array(reaction_rows) * self.size + reaction_cols)).astype(np.int64)
        locations, inverse = np.unique(linear, return_inverse=True)
        constant = np.zeros(len(locations))
        np.add.at(constant, inverse[:decay.nnz], decay.data)
        for name, value, dtype in (("rows", locations // self.size, torch.int64),
                                   ("columns", locations % self.size, torch.int64),
                                   ("constant", constant, torch.float64),
                                   ("positions", inverse[decay.nnz:], torch.int64),
                                   ("weights", weights, torch.float64),
                                   ("parameters_flat", parameters, torch.int64)):
            self.register_buffer(name, torch.tensor(value, dtype=dtype))

    def matrix_values(self, rates):
        """Assemble stored sparse depletion coefficients for every fuel zone.

        Parameters
        ----------
        rates : torch.Tensor
            Nonnegative CPU float64 reaction frequencies in s⁻¹, shaped (zones, rate_nuclides,
            reactions).

        Returns
        -------
        torch.Tensor
            Zone-by-sparse-entry matrix coefficients in s⁻¹.

        Raises
        ------
        ValueError
            Rate axes, precision, device or nonnegative finite values are invalid.
        """
        if rates.shape[-2:] != (len(self.rate_nuclides), len(self.reactions)) or rates.ndim != 3:
            raise ValueError("Rates require zone, named parent and named reaction axes")
        if rates.dtype != torch.float64 or rates.device.type != "cpu" or not torch.isfinite(rates).all() or (rates < 0).any():
            raise ValueError("Rates must be finite, nonnegative CPU float64 frequencies")
        contributions = rates.flatten(1)[:, self.parameters_flat] * self.weights
        return self.constant.expand(len(rates), -1).index_add(1, self.positions, contributions)

    def forward(self, inventory, rates, seconds):
        """Propagate zone inventories with sparse CRAM48 and first derivatives.

        Parameters
        ----------
        inventory : torch.Tensor
            Nonnegative CPU float64 inventory, shaped (zones, full_chain_nuclides).
        rates : torch.Tensor
            Nonnegative CPU float64 reaction frequencies in s⁻¹, shaped (zones, rate_nuclides,
            reactions).
        seconds : float or torch.Tensor
            Nonnegative scalar propagation duration in seconds.

        Returns
        -------
        torch.Tensor
            Final inventories in the same unit as the input, with only roundoff-scale negative
            values clamped to zero.

        Raises
        ------
        ValueError
            Inventory or duration inputs are invalid.
        ArithmeticError
            The propagated inventory is nonfinite or materially negative.
        """
        if inventory.shape != (len(rates), self.size) or inventory.dtype != torch.float64 or inventory.device.type != "cpu":
            raise ValueError("Inventory requires CPU float64 zone and full-chain axes")
        if not torch.isfinite(inventory).all() or (inventory < 0).any():
            raise ValueError("Inventories must be finite and nonnegative")
        seconds = torch.as_tensor(seconds, dtype=torch.float64)
        if seconds.ndim or not torch.isfinite(seconds) or seconds < 0:
            raise ValueError("Duration must be one finite nonnegative scalar in seconds")
        raw = _SparseCram.apply(rates, inventory, seconds, self)
        allowance = 64 * torch.finfo(torch.float64).eps * inventory.detach().max()
        if not torch.isfinite(raw).all() or bool(raw.detach().min() < -allowance):
            raise ArithmeticError("Sparse CRAM returned nonfinite or materially negative density")
        return raw.clamp_min(0.)


class _SparseCram(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rates, inventory, seconds, chain):
        """Apply native sparse CRAM48 while retaining compact backward inputs.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context retaining compact rate, inventory and duration checkpoints.
        rates : torch.Tensor
            Nonnegative CPU float64 reaction frequencies in s⁻¹, shaped (zones, rate_nuclides,
            reactions).
        inventory : torch.Tensor
            Nonnegative CPU float64 inventory, shaped (zones, full_chain_nuclides).
        seconds : float or torch.Tensor
            Nonnegative scalar propagation duration in seconds.
        chain : SparseChain
            Compiled sparse chain connectivity and reaction-to-matrix coefficients.

        Returns
        -------
        torch.Tensor
            Unclipped propagated inventory in the input unit.
        """
        entries, initial = chain.matrix_values(rates).detach().numpy(), inventory.detach().numpy()
        row, col, dt = chain.rows.numpy(), chain.columns.numpy(), float(seconds.detach())
        output = np.empty_like(initial)
        for zone, atoms in enumerate(initial):
            matrix = csc_array((entries[zone], (row, col)), shape=(len(atoms), len(atoms)))
            matrix.eliminate_zeros()
            output[zone] = CRAM48(matrix, atoms, dt)
        ctx.save_for_backward(rates, inventory, seconds)
        ctx.chain = chain
        return torch.from_numpy(output)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        """Reconstruct sparse pole solves for a first-order depletion adjoint.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context retaining compact rate, inventory and duration checkpoints.
        gradient : torch.Tensor
            Incoming objective cotangent with the final inventory shape.

        Returns
        -------
        tuple
            Gradients for rates, initial inventories and elapsed seconds, followed by None for
            static chain metadata.
        """
        rates, inventory, seconds = ctx.saved_tensors
        chain = ctx.chain
        entries = chain.matrix_values(rates).detach().numpy()
        initial, adjoint = inventory.detach().numpy(), gradient.detach().numpy()
        row, col, dt = chain.rows.numpy(), chain.columns.numpy(), float(seconds.detach())
        grad_entries, grad_initial = np.zeros_like(entries), np.empty_like(initial)
        grad_dt = 0.
        identity = eye_array(initial.shape[1], format="csc")
        for zone, atoms in enumerate(initial):
            matrix = csc_array((dt * entries[zone], (row, col)), shape=identity.shape)
            matrix.eliminate_zeros()
            # Only this zone's factors and pole states are retained at once.
            poles, y = [], atoms.copy()
            for alpha, theta in zip(Cram48Solver.alpha, Cram48Solver.theta):
                factor = splu(matrix - theta * identity)
                z = factor.solve(y)
                y += 2 * np.real(alpha * z)
                poles.append((alpha, factor, z))
            g = Cram48Solver.alpha0 * adjoint[zone]
            matrix_gradient = np.zeros(entries.shape[1])
            for alpha, factor, z in reversed(poles):
                # Real-valued input/output: bilinear complex transpose, not
                # Hermitian transpose, gives 2 Re(alpha w_i z_j).
                w = factor.solve(g.astype(np.complex128), trans="T")
                matrix_gradient -= 2 * np.real(alpha * w[row] * z[col])
                g = g + 2 * np.real(alpha * w)
            grad_initial[zone] = g
            grad_entries[zone] = dt * matrix_gradient
            grad_dt += np.dot(matrix_gradient, entries[zone])
        contributions = torch.from_numpy(grad_entries)[:, chain.positions] * chain.weights
        grad_rates = torch.zeros_like(rates).flatten(1).index_add(1, chain.parameters_flat, contributions)
        return (grad_rates.reshape_as(rates), torch.from_numpy(grad_initial),
                torch.tensor(grad_dt, dtype=torch.float64), None)


class SparseCECMDepleter(CECMDepleter):
    """Reuse the checked event/CECM interface with the full sparse OpenMC chain."""
    def __init__(self, compiled, volumes_cm3, *, criticality_tolerance, allow_subcritical_trials=False):
        """Bind the sparse full-chain propagator to the checked CE/CM interface.

        Parameters
        ----------
        compiled : SparseChain
            Compiled OpenMC decay and neutron-reaction connectivity.
        volumes_cm3 : array_like
            One positive physical volume in cm³ per independent fuel state.
        criticality_tolerance : float
            Allowed absolute deviation from the target effective multiplication factor.
        allow_subcritical_trials : bool, optional
            Permit explicitly labelled zero-boron subcritical design trajectories.
        """
        self.compiled = compiled
        self.names, self.rate_nuclides, self.reactions = compiled.names, compiled.rate_nuclides, compiled.reactions
        self.inventory_kind = "atoms_per_cm3"
        self.volumes_cm3 = torch.as_tensor(volumes_cm3, dtype=torch.float64)
        if self.volumes_cm3.ndim != 1 or not torch.isfinite(self.volumes_cm3).all() or (self.volumes_cm3 <= 0).any():
            raise ValueError("Each independent fuel state needs a finite positive volume")
        self.settings = DepletionSettings(criticality_tolerance=criticality_tolerance, max_nuclides=compiled.size,
                                          checkpoint_cram=False, allow_subcritical_trials=allow_subcritical_trials)
        # The shared result field measures dense matrix storage. This adapter
        # uses sparse matrices and compact rate/inventory backward checkpoints.
        self.estimated_dense_memory_bytes = 0

    def _check_memory(self):
        """Report that the sparse adapter allocates no dense chain matrices.

        Returns
        -------
        int
            Zero dense-matrix bytes; this is not a bound on total sparse working memory.
        """
        return 0

    def _advance(self, inventory, rates, duration):
        """Propagate and validate one constant-rate sparse inventory interval.

        Parameters
        ----------
        inventory : torch.Tensor
            Nonnegative CPU float64 inventory, shaped (zones, full_chain_nuclides).
        rates : torch.Tensor
            Nonnegative CPU float64 reaction frequencies in s⁻¹, shaped (zones, rate_nuclides,
            reactions).
        duration : torch.Tensor
            Scalar interval duration in seconds.

        Returns
        -------
        torch.Tensor
            Validated final atom densities in atoms/cm³.
        """
        result = self.compiled(inventory, rates, duration)
        self._check_inventory(result, "depleted inventory")
        return result
