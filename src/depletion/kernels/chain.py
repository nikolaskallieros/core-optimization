"""Static connectivity and tensor assembly from OpenMC Chain.form_matrix.

Source: https://github.com/openmc-dev/openmc/blob/27e38e894697bb32a1dac7848d2618818b6b8daf/openmc/deplete/chain.py at
27e38e894697bb32a1dac7848d2618818b6b8daf (MIT; LICENSE in this directory).
Original XML/Nuclide objects and reaction metadata are reused outside autograd.
Rows are products, columns parents. Duplicate contributions accumulate; each
reaction loss is counted once per parent, irrespective of its product branches.
"""
import math
import torch


class ChainMatrix:
    """Compile an original OpenMC Chain into differentiable dense assembly.

    ``rates[..., nuclide, reaction]`` contains per-parent coefficients in 1/s.
    Rate ordering is explicit; missing products/reaction columns fail at setup.
    Decay, branching and fission yields are static. Optional custom yields have
    the same parent/product mapping as the original ``form_matrix`` argument.
    Transfer/removal, external source and redox terms are outside this adapter.
    """

    def __init__(self, chain, rate_nuclides=None, reactions=None, fission_yields=None):
        """Compile evaluated decay and reaction connectivity for dense assembly.

        Parameters
        ----------
        chain : openmc.deplete.Chain
            Evaluated chain supplying decay constants, branching, reaction connectivity and
            fission yields.
        rate_nuclides : sequence of str or None, optional
            Reaction-rate parent ordering; None uses the full chain order.
        reactions : sequence of str or None, optional
            Reaction-rate column ordering; None uses the chain's reaction list.
        fission_yields : mapping or None, optional
            Parent-to-product yield mapping; None uses the chain's default yields.

        Raises
        ------
        ValueError
            Nuclide/reaction identities are duplicated or decay metadata are inconsistent.
        """
        from openmc.deplete.chain import REACTIONS
        self.names = tuple(nuc.name for nuc in chain.nuclides)
        self.rate_nuclides = tuple(self.names if rate_nuclides is None else rate_nuclides)
        self.reactions = tuple(chain.reactions if reactions is None else reactions)
        if len(set(self.names)) != len(self.names) or len(set(self.rate_nuclides)) != len(self.rate_nuclides):
            raise ValueError('Duplicate nuclide names')
        if len(set(self.reactions)) != len(self.reactions):
            raise ValueError('Duplicate reaction names')
        n = len(self.names)
        indices = {name: i for i, name in enumerate(self.names)}
        rate_indices = {name: i for i, name in enumerate(self.rate_nuclides)}
        rx_indices = {name: i for i, name in enumerate(self.reactions)}
        yields = chain.get_default_fission_yields() if fission_yields is None else fission_yields
        fixed = torch.zeros((n, n), dtype=torch.float64)
        locations, coefficients, rate_locations = [], [], []

        def add(row, col, coefficient, rate):
            """Record one rate-dependent contribution to the depletion matrix.

            Parameters
            ----------
            row : int
                Product nuclide row index.
            col : int
                Parent nuclide column index.
            coefficient : float
                Loss, branching or fission-yield coefficient multiplying the rate.
            rate : int
                Flattened parent-by-reaction input index.
            """
            locations.append(row * n + col)
            coefficients.append(coefficient)
            rate_locations.append(rate)

        for i, nuc in enumerate(chain.nuclides):
            decay = 0.0
            if nuc.half_life is not None:
                if nuc.half_life <= 0:
                    raise ValueError('Nonpositive half life: ' + nuc.name)
                decay = math.log(2) / nuc.half_life
                fixed[i, i] -= decay
            if nuc.n_decay_modes and nuc.half_life is None:
                raise ValueError('Decay modes without half life: ' + nuc.name)
            for mode, target, branch in nuc.decay_modes:
                if target is not None:
                    fixed[indices[target], i] += decay * branch
                if 'alpha' in mode and 'He4' in indices:
                    fixed[indices['He4'], i] += mode.count('alpha') * decay * branch
                elif 'p' in mode and 'H1' in indices:
                    fixed[indices['H1'], i] += mode.count('p') * decay * branch
            if nuc.name not in rate_indices:
                continue
            losses = set()
            for reaction, target, _, branch in nuc.reactions:
                rate = rate_indices[nuc.name] * len(self.reactions) + rx_indices[reaction]
                if reaction not in losses:
                    add(i, i, -1.0, rate)
                    losses.add(reaction)
                if reaction == 'fission':
                    for product, value in yields[nuc.name].items():
                        add(indices[product], i, value, rate)
                else:
                    if target is not None:
                        add(indices[target], i, branch, rate)
                    for light in REACTIONS[reaction].secondaries:
                        if light in indices:
                            add(indices[light], i, branch, rate)
        self._fixed = fixed
        self._locations = torch.tensor(locations, dtype=torch.int64)
        self._coefficients = torch.tensor(coefficients, dtype=torch.float64)
        self._rate_locations = torch.tensor(rate_locations, dtype=torch.int64)

    def __call__(self, rates):
        """Assemble differentiable dense depletion matrices from live rates.

        Parameters
        ----------
        rates : torch.Tensor
            CPU float64 per-parent reaction frequencies in s⁻¹, with leading zone axes followed
            by the declared rate-nuclide and reaction axes.

        Returns
        -------
        torch.Tensor
            Batched square chain matrices in s⁻¹, with product rows and parent columns.

        Raises
        ------
        TypeError
            Rates are not float64 Torch tensors.
        ValueError
            Rate device, axes or finite-value checks fail.
        """
        if rates.dtype != torch.float64:
            raise TypeError('Reaction coefficients must be float64 tensors')
        if rates.device.type != 'cpu':
            raise ValueError('Only CPU depletion is implemented')
        if rates.shape[-2:] != (len(self.rate_nuclides), len(self.reactions)):
            raise ValueError('Rate shape does not match its declared ordering')
        if not torch.isfinite(rates).all():
            raise ValueError('Nonfinite reaction coefficient')
        n = len(self.names)
        # Keep zero-valued edges: the native sparse zero omission must not erase
        # the derivative of a rate that happens to vanish at the current state.
        values = rates.flatten(-2).index_select(-1, self._rate_locations) * self._coefficients
        flat = self._fixed.flatten().expand(*rates.shape[:-2], n * n)
        return flat.index_add(-1, self._locations, values).reshape(*rates.shape[:-2], n, n)
