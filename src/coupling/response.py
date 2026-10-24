"""State-local diffusion responses for controlled, differentiable fuel histories."""
import numpy as np
import torch
from torch.autograd.function import once_differentiable
from optimization.criticality import critical_parameter
from diffusion_surrogate.solver import tensor


class ResponsePoint:
    """Match native values at the calibration point and propagate local changes.

    Physical two-group flux changes scale native fine-spectrum reaction rates.
    Composition and boron changes enter the measured microscopic material
    tables. Correction factors and within-group spectra stay fixed locally.
    """
    def __init__(self, model, state, chain, *, anchor_native=True):
        """Bind a local diffusion response to one measured native state.

        Parameters
        ----------
        model : RegionalSurrogate
            Differentiable regional diffusion model initialized at the native design.
        state : mapping
            Retained native case, transport results, full-chain atom densities and nuclide
            order.
        chain : openmc.deplete.Chain
            Evaluated decay/reaction chain supplying fission energy release and nuclide
            identities.
        anchor_native : bool, optional
            Evaluate and retain native/model ratio anchors; subclasses using absolute responses
            may disable these anchors.

        Raises
        ------
        ValueError
            The design differs from the native anchor or fuel flux/power is nonpositive.
        ArithmeticError
            Rate-derived native fission heat differs from tallied power by more than one
            percent.
        """
        self.model, self.state = model, state
        self.nuclides = tuple(state["nuclides"])
        self.reference_inventory = tensor(state["fuel_inventory_atoms_cm3"])
        self.reference_boron = state["case"].get("boron_mg_B10_per_kg", 0.)
        self.native_keff = state["result"]["combined_keff"][0]
        self.native_rates_by_group = tensor(state["result"]["depletion_response"]["per_parent_s_by_group"])
        self.native_power = tensor(state["result"]["observables"]["region_power_W"]["mean"])[model.fuel_indices]
        self.native_flux = tensor(state["result"]["depletion_response"]["mean_group_flux_cm2_s"])
        self.parent_indices = [self.nuclides.index(n) for n in state["case"]["depletion_nuclides"]]
        self.fission_index = state["case"]["depletion_reactions"].index("fission")
        self.fission_q_j = tensor([sum(r.Q for r in chain.nuclides[chain.nuclide_dict[n]].reactions if r.type == "fission")
                                  for n in state["case"]["depletion_nuclides"]]) * 1.602176634e-19
        self.model.log_factors.requires_grad_(False)
        if any(not np.allclose(getattr(model.design, name).detach().numpy(), value, rtol=0, atol=1e-12)
               for name, value in state["case"]["design"].items()):
            raise ValueError("Construct point anchors at their native design before varying parameters")
        if anchor_native:
            with torch.no_grad():
                nominal = model(self.material_delta(self.reference_inventory, tensor(self.reference_boron)))
                self.anchor_keff = nominal["keff"].clone()
                self.anchor_flux = self.flux(nominal).clone()
                self.anchor_power = nominal["power_W"][model.fuel_indices].clone()
            if (self.anchor_flux <= 0).any() or (self.anchor_power <= 0).any():
                raise ValueError("Every fuel state needs positive native/model flux and power")
        with torch.no_grad():
            native_rates = self.native_rates_by_group.sum(-1)
            raw_heat = self.fission_heat(self.reference_inventory, native_rates)
            self.heat_correction = self.native_power / raw_heat
        if not torch.isfinite(self.heat_correction).all() or (self.heat_correction - 1).abs().max() > .01:
            raise ArithmeticError("Native fission-rate energy differs from its power tally by more than 1%")

    def fission_heat(self, inventory, rates):
        """Compute fuel-region fission heat using live volumes and inventory.

        Parameters
        ----------
        inventory : torch.Tensor
            CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
        rates : torch.Tensor
            Region-by-parent-by-reaction frequencies in s⁻¹, ordered as the native case.

        Returns
        -------
        torch.Tensor
            Unnormalized fission power in W per represented fuel region.
        """
        return (inventory[:, self.parent_indices] * rates[:, :, self.fission_index]
                * self.fission_q_j).sum(-1) * self.physical_volumes()

    def material_delta(self, inventory, boron):
        """Combine live fuel-composition and boron material changes.

        Parameters
        ----------
        inventory : torch.Tensor
            CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
        boron : float or torch.Tensor
            Boron-10 concentration in mg per kg of unborated moderator.

        Returns
        -------
        dict of str to torch.Tensor
            Macroscopic material changes passed to the regional diffusion model.
        """
        fuel = self.model.inventory_material_delta(inventory, self.nuclides)
        control = self.model.boron_material_delta(boron)
        return {name: fuel[name] + control[name] for name in fuel}

    def flux(self, output):
        """Convert integrated diffusion flux to physical fuel-group flux.

        Parameters
        ----------
        output : mapping
            Diffusion outputs containing integrated region flux, source rate and region volumes.

        Returns
        -------
        torch.Tensor
            Fuel-region-by-group flux in n/cm²/s.
        """
        fuel = self.model.fuel_indices
        return output["region_flux_per_source"][fuel] * output["source_rate_per_s"] / output["volume_cm3"][fuel, None]

    def physical_volumes(self):
        """Return live represented fuel-region volumes.

        Returns
        -------
        torch.Tensor
            Volumes in cm³ in native fuel-region order.
        """
        return self.model.fuel_volumes()

    def evaluate(self, inventory, boron):
        """Evaluate a locally anchored response with one power normalization.

        Parameters
        ----------
        inventory : torch.Tensor
            CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
        boron : float or torch.Tensor
            Boron-10 concentration in mg per kg of unborated moderator.

        Returns
        -------
        dict
            Differentiable keff, reaction frequencies in s⁻¹, zone/total powers in W, boron
            concentration and group flux in n/cm²/s.
        """
        output = self.model(self.material_delta(inventory, boron))
        flux = self.flux(output)
        rates = (self.native_rates_by_group * (flux / self.anchor_flux)[:, None, None, :]).sum(-1)
        # One fission-energy normalization governs both depletion and burnup.
        # The fixed native tally residual is tiny and is recorded explicitly.
        heat = self.fission_heat(inventory, rates) * self.heat_correction
        normalization = self.model.power_W / heat.sum()
        power, rates = heat * normalization, rates * normalization
        flux = self.native_flux * flux / self.anchor_flux * normalization
        return {"keff": output["keff"] * self.native_keff / self.anchor_keff, "rates": rates,
                "zone_power_w": power, "power_w": tensor(self.model.power_W), "boron_ppm": tensor(boron),
                "boron_basis": "mg B10 per kg unborated moderator", "group_flux_cm2_s": flux,
                "rate_power_normalization": normalization, "fission_energy_correction": self.heat_correction,
                "nuclide_order": self.nuclides, "reaction_order": tuple(self.state["case"]["depletion_reactions"]),
                "volumes_cm3": self.physical_volumes()}

    def critical(self, inventory, initial=None, bounds=(0., 20.), *, tolerance=1e-10, target=1.):
        """Solve critical boron and retain its implicit first derivatives.

        Parameters
        ----------
        inventory : torch.Tensor
            CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
        initial : float or None, optional
            Initial boron-10 concentration in mg/kg; None starts from the retained native value.
        bounds : tuple of float, optional
            Lower and upper boron-10 concentrations in mg/kg.
        tolerance : float, optional
            Allowed absolute residual in effective multiplication factor at the root.
        target : float, optional
            Target effective multiplication factor.

        Returns
        -------
        dict
            Differentiable keff, reaction frequencies in s⁻¹, zone/total powers in W, boron
            concentration and group flux in n/cm²/s.

        Raises
        ------
        ArithmeticError
            The local boron response does not decrease keff.

        Notes
        -----
        The converged forward boron value is retained. A zero-valued autograd correction
        supplies dB/dx = -k_x/k_B for inventory and design changes.
        """
        enabled = torch.is_grad_enabled()
        root = critical_parameter(lambda b: self.evaluate(inventory.detach(), b)["keff"],
                                  self.reference_boron if initial is None else initial, bounds,
                                  tolerance=tolerance, target=target)
        if root["derivative"] >= 0:
            raise ArithmeticError("Critical boron response must decrease keff")
        boron = tensor(root["parameter"])
        if enabled:
            k = self.evaluate(inventory, boron)["keff"]
            # Zero-valued correction retains the converged root in forward and
            # supplies dB/dx=-k_x/k_B, including inventory and design changes.
            boron = boron - (k - k.detach()) / root["derivative"]
        result = self.evaluate(inventory, boron)
        result["critical_root"] = root
        result["control_status"] = "critical"
        result["criticality_deficit"] = torch.clamp_min(target - result["keff"], 0.)
        return result

    def operate(self, inventory, initial=None, bounds=(0., 20.), *, tolerance=1e-10,
                target=1., allow_subcritical_trials=False, beginning_of_cycle=False):
        """Control critical operation or an explicitly allowed trial deficit.

        Parameters
        ----------
        inventory : torch.Tensor
            CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
        initial : float or None, optional
            Initial boron-10 concentration in mg/kg; None starts from the retained native value.
        bounds : tuple of float, optional
            Lower and upper boron-10 concentrations in mg/kg.
        tolerance : float, optional
            Allowed absolute residual in effective multiplication factor at the root.
        target : float, optional
            Target effective multiplication factor.
        allow_subcritical_trials : bool, optional
            Permit a labelled zero-boron subcritical state after beginning of cycle.
        beginning_of_cycle : bool, optional
            Require critical control even when subcritical trials are otherwise allowed.

        Returns
        -------
        dict
            Differentiable keff, reaction frequencies in s⁻¹, zone/total powers in W, boron
            concentration and group flux in n/cm²/s.

        Notes
        -----
        The active zero-boron bound has dB/dx=0. The actual keff derivative remains available to
        restore feasibility; no negative boron is used.
        """
        if allow_subcritical_trials and not beginning_of_cycle:
            if bounds[0] != 0:
                raise ValueError("Subcritical trials require a zero-boron lower bound")
            with torch.no_grad():
                unborated = self.evaluate(inventory.detach(), tensor(0.))["keff"]
            if unborated < target:
                result = self.evaluate(inventory, tensor(0.))
                result.update(control_status="subcritical_trial",
                              criticality_deficit=torch.clamp_min(target - result["keff"], 0.),
                              critical_root=None)
                return result
        return self.critical(inventory, initial, bounds, tolerance=tolerance, target=target)

    def checkpoint_operate(self, inventory, **options):
        """Evaluate operation with first-order recomputation during backward.

        Parameters
        ----------
        inventory : torch.Tensor
            CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
        **options : object
            Keyword arguments forwarded to operate, including boron bounds and trial policy.

        Returns
        -------
        dict
            Differentiable keff, reaction frequencies in s⁻¹, zone/total powers in W, boron
            concentration and group flux in n/cm²/s.
        """
        return checkpoint_operate(self, inventory, **options)


class _Operation(torch.nn.Module):
    def __init__(self, point, options):
        """Expose a response operation as a functional Torch module.

        Parameters
        ----------
        point : ResponsePoint
            Fixed local transport response with live design parameters.
        options : mapping
            Boron root and trial-policy options passed to the response point.
        """
        super().__init__()
        self.model, self.point, self.options = point.model, point, options

    def forward(self, inventory, initial=None):
        """Evaluate the configured point with an optional warm-start boron.

        Parameters
        ----------
        inventory : torch.Tensor
            CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
        initial : float or None, optional
            Initial boron-10 concentration in mg/kg; None starts from the retained native value.

        Returns
        -------
        dict
            Differentiable keff, reaction frequencies in s⁻¹, zone/total powers in W, boron
            concentration and group flux in n/cm²/s.
        """
        options = dict(self.options)
        if initial is not None:
            options['initial'] = initial
        return self.point.operate(inventory, **options)


class _Recompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, operation, names, inventory, *parameters):
        """Evaluate without retaining solver factors and keep compact inputs.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context retaining inputs and non-tensor operation metadata.
        operation : _Operation
            Functional response operation to recompute during backward.
        names : sequence of str
            Qualified module names matching the supplied design parameter tensors.
        inventory : torch.Tensor
            CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
        *parameters : torch.Tensor
            Live design parameter tensors in names order.

        Returns
        -------
        tuple
            Tensor outputs followed by their keys and detached non-tensor metadata.
        """
        ctx.operation, ctx.names = operation, names
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(inventory, *parameters)
        with torch.no_grad():
            output = torch.func.functional_call(operation, dict(zip(names, parameters)), (inventory,))
        ctx.keys = tuple(k for k, value in output.items() if isinstance(value, torch.Tensor))
        ctx.boron = float(output['boron_ppm'])
        metadata = {k: value for k, value in output.items() if k not in ctx.keys}
        return *(output[k] for k in ctx.keys), (ctx.keys, metadata)

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        """Recompute the response and return first-order input gradients.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context retaining inputs and non-tensor operation metadata.
        *gradients : torch.Tensor or None
            Cotangents of tensor outputs followed by the metadata placeholder.

        Returns
        -------
        tuple
            None for operation/names metadata, then inventory and design parameter gradients.
        """
        needed = ctx.needs_input_grad[2:]
        inputs = [value.detach().requires_grad_(requires)
                  for value, requires in zip(ctx.saved_tensors, needed)]
        with torch.enable_grad():
            output = torch.func.functional_call(ctx.operation, dict(zip(ctx.names, inputs[1:])),
                                                 (inputs[0], ctx.boron))
            pairs = [(output[k], g) for k, g in zip(ctx.keys, gradients[:-1])
                     if g is not None and output[k].requires_grad]
            selected = [value for value in inputs if value.requires_grad]
            if pairs:
                values, weights = zip(*pairs)
                computed = iter(torch.autograd.grad(values, selected, grad_outputs=weights, allow_unused=True))
                result = [next(computed) if requires else None for requires in needed]
            else:
                result = [None] * len(inputs)
        return None, None, *result


def checkpoint_operate(point, inventory, **options):
    """Checkpoint local operation while preserving first derivatives.

    Parameters
    ----------
    point : ResponsePoint
        Fixed local transport response with live design parameters.
    inventory : torch.Tensor
        CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
    **options : object
        Keyword arguments forwarded to ResponsePoint.operate.

    Returns
    -------
    dict
        Differentiable keff, reaction frequencies in s⁻¹, zone/total powers in W, boron
        concentration and group flux in n/cm²/s.

    Notes
    -----
    Closure factors and native observations must remain fixed until backward. Unlike
    tensor-only checkpointing, the forward solver's SciPy factors and Python contexts are
    also released. No OpenMC calculation is replayed.
    """
    if not torch.is_grad_enabled():
        return point.operate(inventory, **options)
    named = dict(point.model.design.named_parameters())
    operation = _Operation(point, options)
    names = tuple('model.design.' + name for name in named)
    values = _Recompute.apply(operation, names, inventory, *named.values())
    keys, metadata = values[-1]
    return {**metadata, **dict(zip(keys, values[:-1]))}
