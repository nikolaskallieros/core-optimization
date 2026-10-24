"""Study CE/CM coupling using the pinned OpenMC-derived chain and CRAM kernels.

The callback owns transport, critical boron and physical power normalization.
It returns per-parent neutron reaction coefficients in 1/s, not reactions/s.
Inventories are either atom totals or atoms/cm3 in explicitly constant volumes.
No chain coefficients or CRAM coefficients are reimplemented here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class DepletionSettings:
    """Declare dense CRAM accuracy, memory limits and powered-stage acceptance."""
    cram_order: int = 48
    target_keff: float = 1.0
    criticality_tolerance: float = 1e-8
    max_nuclides: int = 128
    max_dense_memory_mib: float = 256.0
    checkpoint_cram: bool = True
    allow_subcritical_trials: bool = False

    def __post_init__(self):
        """Validate CRAM order, dense limits and powered-stage controls.

        Raises
        ------
        ValueError
            A setting has an unsupported value or type.
        """
        if self.cram_order not in (16, 48):
            raise ValueError('CRAM order must be 16 or 48')
        if isinstance(self.max_nuclides, bool) or not isinstance(self.max_nuclides, int) or self.max_nuclides < 1:
            raise ValueError('max_nuclides must be a positive integer')
        for name in ('target_keff', 'criticality_tolerance', 'max_dense_memory_mib'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not isinstance(self.allow_subcritical_trials, bool):
            raise ValueError('allow_subcritical_trials must be boolean')


class DepletionError(RuntimeError):
    """A failed integration/physical-stage check; partial results are not accepted."""


@dataclass(frozen=True)
class DepletionStage:
    """Keep a transport response with the exact inventory and time it evaluated."""
    name: str
    time_seconds: torch.Tensor
    inventory: torch.Tensor
    state: Mapping[str, Any]


@dataclass(frozen=True)
class DepletionStep:
    """Return the CE/CM endpoint and its beginning, midpoint and end evaluations."""
    inventory: torch.Tensor
    end_time_seconds: torch.Tensor
    stages: tuple[DepletionStage, ...]
    inventory_kind: str
    volumes_cm3: torch.Tensor
    estimated_dense_memory_bytes: int

    @property
    def atom_totals(self):
        """Return total atoms using the declared inventory convention.

        Returns
        -------
        torch.Tensor
            Zone-by-chain atom totals, multiplying densities by constant zone volumes when
            needed.
        """
        if self.inventory_kind == 'atoms':
            return self.inventory
        return self.inventory * self.volumes_cm3[..., None]


def _load_pinned_kernels():
    """Load the bundled OpenMC-derived chain assembly and CRAM kernels.

    Returns
    -------
    tuple
        ChainMatrix factory and the differentiable dense CRAM callable.
    """
    from depletion.kernels import ChainMatrix, cram
    return ChainMatrix, cram


def _cpu_float64(value, name):
    """Require a finite CPU float64 tensor without detaching its graph.

    Parameters
    ----------
    value : object
        Numerical value to validate or convert.
    name : str
        Field or stage label used in validation diagnostics.

    Returns
    -------
    torch.Tensor
        The validated input tensor.

    Raises
    ------
    TypeError
        The input is not a CPU float64 tensor.
    DepletionError
        The tensor contains nonfinite values.
    """
    if value.dtype != torch.float64 or value.device.type != 'cpu':
        raise TypeError(f'{name} must be a CPU float64 tensor')
    if not torch.isfinite(value).all():
        raise DepletionError(f'{name} must be finite')
    return value


def _scalar(value, name):
    """Convert a finite scalar while preserving an existing Torch graph.

    Parameters
    ----------
    value : object
        Numerical value to validate or convert.
    name : str
        Field or stage label used in validation diagnostics.

    Returns
    -------
    torch.Tensor
        Zero-dimensional CPU float64 tensor.

    Raises
    ------
    DepletionError
        The value is nonfinite or nonscalar.
    """
    if isinstance(value, torch.Tensor):
        tensor = _cpu_float64(value, name)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float64, device='cpu')
    if tensor.ndim or not torch.isfinite(tensor):
        raise DepletionError(f'{name} must be a finite scalar')
    return tensor


def admissible_powered_state(state, settings):
    """Check critical operation or explicitly permitted subcritical trials.

    Parameters
    ----------
    state : mapping
        Stage state with keff, nonnegative boron_ppm and optional control_status.
    settings : DepletionSettings
        Accuracy, memory and powered-stage admissibility controls.

    Returns
    -------
    bool
        Whether the state meets criticality tolerance or the enabled zero-boron
        subcritical-trial contract.

    Notes
    -----
    Subcritical trials retain their actual eigenvalue and use zero boron. Their
    fission-power-normalized depletion is an optimization trajectory, not a physically
    self-sustained operating history.
    """
    if state['keff'] is None or state['boron_ppm'] is None:
        return False
    k, boron = (_scalar(state[name], name) for name in ('keff', 'boron_ppm'))
    if boron < 0:
        return False
    if abs(k.item() - settings.target_keff) <= settings.criticality_tolerance:
        return True
    return (getattr(settings, 'allow_subcritical_trials', False)
            and state.get('control_status') == 'subcritical_trial'
            and boron.item() == 0 and 0 < k.item() < settings.target_keff)


class CECMDepleter:
    """One- or many-zone midpoint depletion in time-constant volumes.

    ``inventory_kind`` is required and is either ``atoms`` or ``atoms_per_cm3``.
    ``volumes_cm3.shape`` defines the batch shape; a scalar volume means one
    unbatched zone. An inventory has that batch shape followed by the complete
    chain nuclide axis. Volumes may depend on design parameters, but do not
    change during a depletion interval. Refuelling/material transfer is handled
    outside this adapter and must split intervals at its events.

    The stage callback receives ``(inventory, time_seconds)`` and returns a
    mapping containing ``rates``, ``power_w``, ``keff`` and ``boron_ppm``.
    Rates have shape [batch..., rate_nuclide, reaction], with explicit ordering
    given by ``rate_nuclides`` and ``reactions``. A powered callback must solve
    its current critical state unless subcritical design trials are explicitly
    enabled and labelled; zero-power callbacks must return zero neutron
    rates and may return None for keff/boron. Remaining metadata is retained.
    The adapter checks outputs but cannot prove that a callback ran transport.

    Only pure CRAM calls are checkpointed: callbacks execute once at beginning,
    midpoint and end and are never replayed by backward. The memory estimate is
    a conservative dense-array gate per interval, not a measured bound on
    transport or an entire retained multistep autograd history.

    Optional kernel injection supports isolated qualification using the actual
    pinned ChainMatrix/CRAM objects, without triggering dependency setup here.
    """

    def __init__(self, chain, *, inventory_kind: str, volumes_cm3,
                 settings: DepletionSettings | None = None,
                 rate_nuclides=None, reactions=None,
                 chain_matrix_factory=None, cram_solver=None):
        """Bind an evaluated chain to constant-volume midpoint depletion.

        Parameters
        ----------
        chain : openmc.deplete.Chain
            Evaluated chain supplying decay constants, branching, reaction connectivity and
            fission yields.
        inventory_kind : {'atoms', 'atoms_per_cm3'}
            Explicit meaning of every inventory vector.
        volumes_cm3 : array_like or torch.Tensor
            Positive constant zone volumes in cm³; shape defines the leading inventory batch
            axes.
        settings : DepletionSettings or None, optional
            Dense-kernel and powered-stage controls; None uses DepletionSettings defaults.
        rate_nuclides : sequence of str or None, optional
            Rate-parent ordering; None uses the full chain order.
        reactions : sequence of str or None, optional
            Rate-reaction ordering; None uses the evaluated chain order.
        chain_matrix_factory : callable or None, optional
            Optional chain-assembly implementation for isolated kernel verification.
        cram_solver : callable or None, optional
            Optional CRAM implementation accepting matrix, inventory, duration and order.

        Raises
        ------
        ValueError
            Inventory convention or constant volumes are invalid.
        DepletionError
            The chain or estimated dense working storage exceeds configured limits.
        """
        self.settings = settings or DepletionSettings()
        self.names = tuple(nuclide.name for nuclide in chain.nuclides)
        if not self.names or len(self.names) > self.settings.max_nuclides:
            raise DepletionError('Chain size is outside the configured dense-kernel limit')
        if inventory_kind not in ('atoms', 'atoms_per_cm3'):
            raise ValueError('inventory_kind must be atoms or atoms_per_cm3')
        self.inventory_kind = inventory_kind
        if isinstance(volumes_cm3, torch.Tensor):
            self.volumes_cm3 = _cpu_float64(volumes_cm3, 'volumes_cm3').clone()
        else:
            self.volumes_cm3 = torch.as_tensor(volumes_cm3, dtype=torch.float64).clone()
        if (not torch.isfinite(self.volumes_cm3).all() or self.volumes_cm3.numel() == 0
                or (self.volumes_cm3 <= 0).any()):
            raise ValueError('Zone volumes must be finite and positive')
        if chain_matrix_factory is None or cram_solver is None:
            factory, solver = _load_pinned_kernels()
            chain_matrix_factory = chain_matrix_factory or factory
            cram_solver = cram_solver or solver
        self.chain_matrix = chain_matrix_factory(chain, rate_nuclides=rate_nuclides, reactions=reactions)
        self.rate_nuclides = tuple(self.chain_matrix.rate_nuclides)
        self.reactions = tuple(self.chain_matrix.reactions)
        self._cram = cram_solver
        self.estimated_dense_memory_bytes = self._check_memory()

    def _check_memory(self):
        """Enforce the estimated dense CRAM working-storage limit.

        Returns
        -------
        int
            Conservative per-interval matrix working-storage estimate in bytes.

        Raises
        ------
        DepletionError
            The estimate exceeds the configured memory budget.
        """
        # Four complex matrix-sized arrays per pole allow for factorization,
        # solve and adjoint working storage. Checkpointing retains one CRAM
        # recomputation plus inputs; uncheckpointed CE/CM can retain both.
        poles = self.settings.cram_order // 2
        simultaneous_stages = 1 if self.settings.checkpoint_cram else 2
        estimate = (self.volumes_cm3.numel() * len(self.names) ** 2 * 16
                    * (4 * poles * simultaneous_stages + 4))
        if estimate > self.settings.max_dense_memory_mib * 1024 ** 2:
            raise DepletionError(f'Dense CRAM estimate {estimate} bytes exceeds configured memory budget')
        return estimate

    def _check_inventory(self, inventory, name):
        """Validate inventory shape, precision, finiteness and nonnegativity.

        Parameters
        ----------
        inventory : torch.Tensor
            Nonnegative CPU float64 inventory with leading zone axes and the full chain-nuclide
            axis, in the adapter's declared inventory unit.
        name : str
            Field or stage label used in validation diagnostics.

        Raises
        ------
        DepletionError
            Inventory values or axes violate the declared constant-volume chain contract.
        """
        _cpu_float64(inventory, name)
        if inventory.shape != (*self.volumes_cm3.shape, len(self.names)):
            raise DepletionError(f'{name} shape does not match constant volumes and chain order')
        if (inventory < 0).any():
            raise DepletionError(f'{name} contains negative inventory; no clipping is applied')

    def _evaluate(self, operator, inventory, time, name):
        """Evaluate and validate a physical depletion-stage callback.

        Parameters
        ----------
        operator : callable
            Evaluate operator(inventory, time_seconds) and return rates, power_w, keff and
            boron_ppm.
        inventory : torch.Tensor
            Nonnegative CPU float64 inventory with leading zone axes and the full chain-nuclide
            axis, in the adapter's declared inventory unit.
        time : torch.Tensor
            Scalar absolute stage time in seconds.
        name : str
            Field or stage label used in validation diagnostics.

        Returns
        -------
        DepletionStage
            Validated stage state tied to its exact inventory and time.

        Raises
        ------
        DepletionError
            The callback mutates inputs or supplies invalid rates, power or criticality
            diagnostics.
        """
        before = inventory._version
        volume_before = self.volumes_cm3._version
        state = operator(inventory, time)
        if inventory._version != before or self.volumes_cm3._version != volume_before:
            raise DepletionError('Stage callback mutated its inventory or constant volumes')
        rates = _cpu_float64(state['rates'], 'rates')
        expected = (*self.volumes_cm3.shape, len(self.rate_nuclides), len(self.reactions))
        if rates.shape != expected or (rates < 0).any():
            raise DepletionError('Rates must be nonnegative per-parent 1/s coefficients in declared order')
        power = _scalar(state['power_w'], 'power_w')
        if power < 0:
            raise DepletionError('Stage power must be nonnegative')
        if state['boron_ppm'] is not None and _scalar(state['boron_ppm'], 'boron_ppm') < 0:
            raise DepletionError('Negative critical boron is infeasible')
        if power == 0:
            if torch.count_nonzero(rates):
                raise DepletionError('Zero-power interval requires zero neutron reaction rates')
        else:
            if not admissible_powered_state(state, self.settings):
                raise DepletionError(f'{name} stage is not critical within the configured tolerance')
        return DepletionStage(name, time, inventory, dict(state))

    def _advance(self, inventory, rates, duration):
        """Propagate an inventory with frozen rates and optional CRAM checkpointing.

        Parameters
        ----------
        inventory : torch.Tensor
            Nonnegative CPU float64 inventory with leading zone axes and the full chain-nuclide
            axis, in the adapter's declared inventory unit.
        rates : torch.Tensor
            CPU float64 per-parent reaction frequencies in s⁻¹, with leading zone axes followed
            by the declared rate-nuclide and reaction axes.
        duration : torch.Tensor
            Scalar interval duration in seconds.

        Returns
        -------
        torch.Tensor
            Depleted inventory in the input unit and chain order.
        """
        matrix = self.chain_matrix(rates)
        def kernel(a, n, dt):
            """Apply the selected dense CRAM order to one frozen operator.

            Parameters
            ----------
            a : torch.Tensor
                Batched depletion matrix in s⁻¹.
            n : torch.Tensor
                Initial inventory in the adapter's declared unit.
            dt : torch.Tensor
                Scalar elapsed time in seconds.

            Returns
            -------
            torch.Tensor
                Propagated inventory without clipping.
            """
            return self._cram(a, n, dt, self.settings.cram_order)
        differentiated = torch.is_grad_enabled() and any(x.requires_grad for x in (matrix, inventory, duration))
        if self.settings.checkpoint_cram and differentiated:
            result = checkpoint(kernel, matrix, inventory, duration,
                                use_reentrant=False, preserve_rng_state=False)
        else:
            result = kernel(matrix, inventory, duration)
        self._check_inventory(result, 'depleted inventory')
        return result

    def step(self, inventory: torch.Tensor, time_seconds, dt_seconds,
             stage_operator: Callable) -> DepletionStep:
        """Advance one CE/CM interval with beginning, midpoint and end solves.

        Parameters
        ----------
        inventory : torch.Tensor
            Nonnegative CPU float64 inventory with leading zone axes and the full chain-nuclide
            axis, in the adapter's declared inventory unit.
        time_seconds : float or torch.Tensor
            Nonnegative absolute starting time in seconds.
        dt_seconds : float or torch.Tensor
            Positive interval duration in seconds.
        stage_operator : callable
            Physical callback receiving the current inventory and time in seconds, returning
            per-parent rates and operating-state diagnostics.

        Returns
        -------
        DepletionStep
            Corrected endpoint inventory, endpoint time and all three evaluated stage records.

        Notes
        -----
        The predictor advances the initial inventory for half the interval. The midpoint rates
        then advance the original inventory for the full interval. Transport callbacks execute
        once per stage; only pure CRAM work may be replayed during backward.
        """
        self._check_inventory(inventory, 'initial inventory')
        time = _scalar(time_seconds, 'time_seconds')
        duration = _scalar(dt_seconds, 'dt_seconds')
        if time < 0 or duration <= 0:
            raise DepletionError('Require nonnegative starting time and positive elapsed seconds')
        start = self._evaluate(stage_operator, inventory, time, 'beginning')
        half_inventory = self._advance(inventory, start.state['rates'], duration / 2)
        middle = self._evaluate(stage_operator, half_inventory, time + duration / 2, 'midpoint')
        # The corrector advances the ORIGINAL inventory over the FULL interval.
        final_inventory = self._advance(inventory, middle.state['rates'], duration)
        end_time = time + duration
        end = self._evaluate(stage_operator, final_inventory, end_time, 'end')
        return DepletionStep(final_inventory, end_time, (start, middle, end),
                             self.inventory_kind, self.volumes_cm3,
                             self.estimated_dense_memory_bytes)


def reaction_weighted_change(before, after, rates, volumes):
    """Measure inventory change using sampled reaction-frequency weights.

    Parameters
    ----------
    before : array_like
        Earlier region-by-nuclide inventories in a consistent density unit.
    after : array_like
        Later inventories with the same shape and unit.
    rates : array_like
        Nonnegative sampled per-parent reaction frequencies in s⁻¹, shaped like the
        inventories.
    volumes : array_like
        Nonnegative region volumes in cm³.

    Returns
    -------
    dict
        Dimensionless global, maximum-region and per-region relative changes.

    Raises
    ------
    ValueError
        Axes or values are invalid, or no positive weighted reaction scale exists.

    Notes
    -----
    This measures the change in reaction rates at a common sampled flux. It is a fixed-point
    stopping diagnostic, not a reactivity or uncertainty estimate. All inventories are
    retained, including nuclides with negligible rates.
    """
    before, after, rates, volumes = map(np.asarray, (before, after, rates, volumes))
    if before.shape != after.shape or before.shape != rates.shape or volumes.shape != before.shape[:1]:
        raise ValueError('Inventory, rate and volume axes differ')
    if not all(np.isfinite(value).all() and (value >= 0).all() for value in (before, after, rates, volumes)):
        raise ValueError('Convergence weights and inventories must be finite and nonnegative')
    change = (abs(after-before)*rates).sum(-1)
    scale = (np.maximum(before, after)*rates).sum(-1)
    if np.dot(scale, volumes) <= 0:
        raise ValueError('No sampled reactions for the inventory convergence check')
    relative = np.divide(change, scale, out=np.zeros_like(change), where=scale > 0)
    return {'global': float(np.dot(change, volumes)/np.dot(scale, volumes)),
            'maximum_region': float(relative.max()), 'by_region': relative.tolist()}
