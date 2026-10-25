"""Small multigroup diffusion/transport histories using the shared CE/CM kernels."""

import torch

from fuel_cycle.residence import cecm
from materials.composition import reaction_rates


class DepletionResponse(torch.nn.Module):
    """Bind regional transport to microscopic depletion rates and physical power."""

    def __init__(self, transport, macroscopic, microscopic_rates, fuel_indices, *, reference=None):
        """Connect a material callback to native or calibrated transport.

        Parameters
        ----------
        transport : callable or torch.nn.Module
            Native material-table evaluator, or CalibratedDiffusion when reference
            is supplied. Returns integrated regional flux, source rate and powers.
        macroscopic : callable
            Map fuel-region inventories in atoms/cm³ to all-region material tables.
        microscopic_rates : torch.Tensor
            Fuel-region/nuclide/reaction/group cross sections in barns, ordered
            consistently with the depletion chain. Corrections to reaction cross
            sections must also be represented here; diffusion-only fits need none.
        fuel_indices : sequence of int
            Fuel region indices in transport's region ordering.
        reference : dict or None, optional
            Physical reference tables held by CalibratedDiffusion. None passes
            absolute material tables to a native evaluator.
        """
        super().__init__()
        self.transport = transport
        object.__setattr__(self, "macroscopic", macroscopic)
        self.reference = reference
        self.register_buffer("microscopic_rates", microscopic_rates)
        self.register_buffer("fuel_indices", torch.as_tensor(fuel_indices, dtype=torch.long))

    def forward(self, inventory):
        """Evaluate flux-dependent reaction frequencies for one fuel state.

        Parameters
        ----------
        inventory : torch.Tensor
            Fuel-region-by-chain number densities in atoms/cm³.

        Returns
        -------
        dict
            Transport responses, per-parent reaction frequencies in s⁻¹, and fuel
            powers in W in the inventory's region ordering.
        """
        macro = self.macroscopic(inventory)
        inputs = macro if self.reference is None else {
            name: value - self.reference[name] for name, value in macro.items()}
        result = self.transport(inputs)
        indices = self.fuel_indices
        flux = (result["region_flux_per_source"][indices] * result["source_rate_per_s"]
                / result["volume_cm3"][indices, None])
        return {**result, "rates": reaction_rates(self.microscopic_rates, flux),
                "zone_power_w": result["power_W"][indices]}


class DepletionHistory(torch.nn.Module):
    """Differentiate a prescribed-power history without changing or moving fuel.

    The eigenvalue is reported at every transport stage. Power normalization
    does not imply that an uncontrolled, noncritical state is an operating reactor.
    """

    def __init__(self, response, chain, initial_state, durations_days):
        """Bind a transport response and fresh-state factory to a short history.

        Parameters
        ----------
        response : DepletionResponse
            Inventory-dependent transport and reaction rates.
        chain : depletion.sparse.SparseChain
            Compiled OpenMC chain with matching isotope and reaction ordering.
        initial_state : callable
            Return a fresh FuelState with live densities, masses and identities.
        durations_days : array_like
            Successive CE/CM interval lengths in days.
        """
        super().__init__()
        self.response, self.chain = response, chain
        object.__setattr__(self, "initial_state", initial_state)
        self.register_buffer("durations_days", torch.as_tensor(durations_days, dtype=torch.float64))

    def forward(self):
        """Evaluate beginning, midpoint and endpoint transport and isotope states.

        Returns
        -------
        dict of torch.Tensor
            Final keff, stage times in days, keff/power/inventory histories, final
            inventories in atoms/cm³ and integrated fuel energy in J. Native
            standard errors describe individual transport calculations only.
        """
        state = self.initial_state()
        start = self.response(state.inventory)
        outputs, inventories = [start], [state.inventory]
        time = self.durations_days.new_zeros(())
        times = [time]
        for duration in self.durations_days:
            state, middle_inventory, middle = cecm(state, duration, start, self.response, self.chain)
            start = self.response(state.inventory)
            outputs.extend((middle, start))
            inventories.extend((middle_inventory, state.inventory))
            times.extend((time + duration / 2, time + duration))
            time = time + duration
        result = {"keff": start["keff"], "time_days": torch.stack(times),
                  "keff_history": torch.stack([output["keff"] for output in outputs]),
                  "power_history_W": torch.stack([output["zone_power_w"] for output in outputs]),
                  "inventory_history": torch.stack(inventories), "inventory": state.inventory,
                  "energy_j": state.energy_j, "initial_hm_kg": state.initial_hm_kg}
        if "standard_error" in start:
            result["standard_error"] = start["standard_error"]
            result["standard_error_history"] = torch.stack([output["standard_error"] for output in outputs])
        return result
