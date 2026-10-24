"""Shared spatial corrections fitted through a connected Bateman trajectory."""
import json
import time

import torch
from diffusion_surrogate.solver import tensor
from runtime.io import sha

from fuel_cycle.residence import replay_history
from materials.composition import reaction_rates
from coupling.history import cycle_point_keys
from coupling.response import ResponsePoint
from runtime.io import load_state


class SharedCoefficients(torch.nn.Module):
    """Bound one trainable vector of log correction factors for all native states."""
    def __init__(self, count, bound):
        """Create one bounded coefficient vector shared by every stage.

        Parameters
        ----------
        count : int
            Number of coefficients in the selected spatial basis.
        bound : float
            Positive absolute bound on dimensionless log correction factors.
        """
        super().__init__()
        self.raw = torch.nn.Parameter(torch.zeros(count, dtype=torch.float64))
        self.bound = float(bound)

    def forward(self):
        """Map trainable raw coefficients to bounded log correction factors.

        Returns
        -------
        torch.Tensor
            Shared log factors bounded smoothly by the configured magnitude.
        """
        return self.bound * torch.tanh(self.raw / self.bound)

    def initialize(self, values):
        """Initialize raw parameters from bounded physical log factors.

        Parameters
        ----------
        values : array_like or torch.Tensor
            Dimensionless log correction coefficients in the declared spatial basis.
        """
        with torch.no_grad():
            fraction = (tensor(values) / self.bound).clamp(-1. + 1e-12, 1. - 1e-12)
            self.raw.copy_(self.bound * torch.atanh(fraction))


def transfer_coefficients(case, source_layout, values, target_layout, bound, *, basis_factory):
    """Transfer corrections only when the target basis represents them.

    Parameters
    ----------
    case : mapping
        Transport case used to construct both spatial operator bases.
    source_layout : str
        Layout defining the input coefficients.
    values : array_like or torch.Tensor
        Dimensionless log correction coefficients in the declared spatial basis.
    target_layout : str
        Layout to receive the same represented operator correction.
    bound : float
        Positive absolute bound on dimensionless log correction factors.
    basis_factory : callable
        Build names, diffusion bases and equivalence bases from case, layout and
        include_void.

    Returns
    -------
    torch.Tensor or None
        Target coefficients when the represented field agrees within 1e-10 and respects the
        bound; otherwise None.
    """
    matrices = []
    for layout in (source_layout, target_layout):
        names, bases, equivalence = basis_factory(case, layout=layout, include_void=False)
        matrices.append(torch.cat((bases["diffusion"].flatten(1), equivalence.flatten(1)), dim=1).T)
    source, target = matrices
    field = source @ tensor(values)
    projected = torch.linalg.lstsq(target, field, driver="gelsd").solution
    if (target @ projected - field).abs().max() > 1e-10 or projected.abs().max() > bound:
        return None
    return projected


class SharedPoint(ResponsePoint):
    """Use absolute diffusion predictions with measured microscopic responses.

    No fitted coefficient depends on the time index. Native spectra provide
    measured group response tables at each state; local composition changes
    enter explicitly. There is no native/model value-ratio correction.
    """
    def __init__(self, model, state, chain, coefficients):
        """Bind absolute microscopic responses to shared correction factors.

        Parameters
        ----------
        model : RegionalSurrogate
            Regional diffusion model with live geometry and measured material responses.
        state : mapping
            Retained native case, transport results, full-chain atom densities and nuclide
            order.
        chain : openmc.deplete.Chain
            Evaluated decay/reaction chain supplying fission energy release and nuclide
            identities.
        coefficients : SharedCoefficients
            One coefficient module reused at every retained time state.
        """
        super().__init__(model, state, chain, anchor_native=False)
        self.coefficients = coefficients
        self._last_coefficients = None
        if not state["case"].get("physical_inventory_response"):
            raise ValueError("Lifetime fitting requires absolute isotope-based material data")
        self.rate_xs_b = tensor(state["result"]["depletion_response"]["microscopic_b"])

    def evaluate(self, inventory, boron):
        """Evaluate absolute shared-factor flux, rates and power.

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

        Notes
        -----
        Rates use measured microscopic cross sections and the predicted group flux. One
        fission-energy normalization sets rates, flux and power consistently; native/model value
        ratios are not used.
        """
        delta = self.material_delta(inventory, boron)
        values = self.coefficients()
        if self._last_coefficients is None or not torch.equal(values.detach(), self._last_coefficients):
            # A global calibration step can change every material at once.
            # Initialize its eigenmode by positive power iteration.
            self.model.solver._guess = None
            self._last_coefficients = values.detach().clone()
        output = torch.func.functional_call(self.model, {"log_factors": values}, (delta,))
        flux = self.flux(output)
        rates = reaction_rates(self.rate_xs_b, flux)
        heat = self.fission_heat(inventory, rates)
        normalization = self.model.power_W / heat.sum()
        power, rates, flux = heat * normalization, rates * normalization, flux * normalization
        return {"keff": output["keff"], "rates": rates, "zone_power_w": power,
                "power_w": tensor(self.model.power_W), "boron_ppm": tensor(boron),
                "group_flux_cm2_s": flux, "diffusion_fields": output}


class SharedCycle(torch.nn.Module):
    """Fit shared spatial factors across native states and their connected depletion history.

    Case factories supply geometry, observations and the response basis; the
    same coefficients are used throughout the complete residence calculation."""
    def __init__(self, root, physical, chain, mapping, entry, history, design, layout):
        """Build shared responses and fitting targets for a full residence.

        Parameters
        ----------
        root : CycleCase
            Resolved case providing model factories, settings and result paths.
        physical : IsotopicInventory
            Full-chain inventory binding, native case factory and live fresh-fuel recipe.
        chain : SparseChain
            Differentiable full-chain propagator accepting inventory, reaction frequencies in
            s⁻¹ and elapsed seconds.
        mapping : FuelMap
            Region-to-channel/axial mapping with physical multiplicities and symmetry fractions.
        entry : FuelState
            Region-ordered inventories, carried energy in J, initial heavy-metal masses in kg
            and physical fuel identities.
        history : mapping
            Retained native cycle schedule, observations, design and ordered stage keys.
        design : torch.nn.Module
            Live case design, including the common cycle_days parameter.
        layout : str
            Case-declared spatial correction-basis layout.
        """
        super().__init__()
        self.workspace = root
        self.physical, self.chain = physical, chain
        self.mapping, self.entry, self.history, self.design = mapping, entry, history, design
        self.config = self.workspace.settings()
        self.fit_config = self.workspace.settings("calibration")
        self.layout, self.keys = layout, cycle_point_keys(history)
        directory = self.workspace.native_states()
        self.points, self.targets, self.references = {}, {}, {}
        models = []
        for key in self.keys:
            path = directory / f"{key}.json"
            state = load_state(path)
            model = self.workspace.model.make_surrogate(design, state, layout)
            if not models:
                self.parameter_names = model.parameter_names
                self.coefficients = SharedCoefficients(len(self.parameter_names), self.fit_config["log_factor_bound"])
            point = SharedPoint(model, state, physical.chain, self.coefficients)
            self.points[key] = point
            targets, reference = self.workspace.model.fitting_targets(state["result"], state["case"])
            self.targets[key] = targets
            self.references[key] = {name: {field: tensor(value) for field, value in entry.items()}
                                    for name, entry in reference["observables"].items()}
            models.append(model)
        self.models = torch.nn.ModuleList(models)
        # Start from physical OpenMC coefficients. All fitted corrections are
        # shared across time; independent per-state fits are neither used nor required.
        maximum = torch.stack([p.reference_inventory.max(0).values for p in self.points.values()]).max(0).values
        self.inventory_mask = maximum >= self.fit_config["inventory_floor_atoms_cm3"]
        self.latest = {}

    def observations(self, key, inventory, output):
        """Compare stage fields, reaction sources and inventories with data.

        Parameters
        ----------
        key : str
            Native state identifier within the retained history.
        inventory : torch.Tensor
            CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
        output : mapping
            Stage response containing diffusion_fields, rates, powers, keff and fuel-group flux.

        Returns
        -------
        loss : torch.Tensor
            Weighted spatial and reaction-source loss.
        relative : dict
            Per-family relative field and reaction-source differences.
        scaled : dict
            Residuals normalized by declared fitting scales.
        inventory_error : torch.Tensor
            Relative inventory error above the configured atom-density floor.
        """
        fields = dict(output["diffusion_fields"])
        fields["fuel_power_W"] = output["zone_power_w"]
        fields["keff"] = output["keff"]
        differences, normalized, losses = {}, {}, []
        for target in self.targets[key]:
            expected = self.references[key][target.name]
            difference = target.values(fields) - expected["mean"]
            residual = difference / expected["scale"]
            losses.append(target.weight * residual.square().sum())
            normalized[target.name] = residual
            floor = expected["mean"].abs().max().clamp_min(1e-30) * 1e-4
            differences[target.name] = difference / expected["mean"].abs().clamp_min(floor)
        point = self.points[key]
        differences["fuel_group_flux"] = ((output["group_flux_cm2_s"] - point.native_flux)
                                           / point.native_flux.abs().clamp_min(1e-30))
        source_groups = (point.reference_inventory[:, point.parent_indices, None, None]
                         * point.native_rates_by_group).sum(1)
        expected_source = source_groups.sum(-1)
        actual_source = (inventory[:, point.parent_indices, None] * output["rates"]).sum(1)
        floor = expected_source.abs().amax(0).clamp_min(1e-30)[None, :] * 1e-4
        relative_source = (actual_source - expected_source) / expected_source.abs().clamp_min(floor)
        flux_tally = point.state["result"]["observables"]["region_flux_per_source"]
        indices = point.model.fuel_indices
        flux_relative_scale = (tensor(flux_tally["scale"])[indices]
                               / tensor(flux_tally["mean"])[indices].abs().clamp_min(1e-30))
        # Use the sum of group scale contributions without assuming independent
        # group errors. These are fitting weights, not a complete rate covariance.
        reaction_scale = torch.maximum((source_groups.abs() * flux_relative_scale[:, None, :]).sum(-1),
            expected_source.abs().clamp_min(floor) * self.fit_config["reaction_relative_scale"])
        normalized["depletion_reactions"] = (actual_source - expected_source) / reaction_scale
        differences["depletion_reactions"] = relative_source
        losses.append(normalized["depletion_reactions"].square().mean())
        reference_n = point.reference_inventory[:, self.inventory_mask]
        inventory_error = ((inventory[:, self.inventory_mask] - reference_n)
                           / reference_n.abs().clamp_min(self.fit_config["inventory_floor_atoms_cm3"]))
        return torch.stack(losses).sum(), differences, normalized, inventory_error

    def forward(self, *, coupled=True):
        """Evaluate pooled or connected trajectory calibration loss.

        Parameters
        ----------
        coupled : bool, optional
            Fit the connected depletion trajectory; False pools responses at fixed native
            inventories.

        Returns
        -------
        torch.Tensor
            Scalar field, inventory and coefficient-regularization loss; detached diagnostics
            are also stored on the module.
        """
        observed, predicted = {}, {}

        def observe(key, inventory, output):
            """Collect stage residuals and detached coupled predictions.

            Parameters
            ----------
            key : str
                Native state identifier within the retained history.
            inventory : torch.Tensor
                CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
            output : mapping
                Stage response containing diffusion_fields, rates, powers, keff and fuel-group flux.
            """
            observed[key] = self.observations(key, inventory, output)
            if coupled:
                predicted[key] = {"power_W": output["zone_power_w"].detach().tolist(),
                                  "group_flux_cm2_s": output["group_flux_cm2_s"].detach().tolist(),
                                  "keff": float(output["keff"].detach())}

        if coupled:
            metrics, final, samples, cycle_days = replay_history(
                self.mapping, self.chain, self.design, self.history, self.points, self.config,
                fixed_cycle_days=self.history["cycles"][-1]["cycle_days"], observe=observe)
        else:
            from diffusion_surrogate.regional import shared_geometry
            with shared_geometry(self.points):
                for key, point in self.points.items():
                    inventory = point.reference_inventory
                    observe(key, inventory, point.evaluate(inventory, tensor(0.)))
        if set(observed) != set(self.keys):
            raise ArithmeticError("The trajectory fit omitted a calibration state")
        spatial = torch.stack([observed[key][0] for key in self.keys]).mean()
        composition = torch.stack([observed[key][3].square().mean() for key in self.keys]).mean()
        regularization = self.fit_config["log_factor_regularization"] * self.coefficients().square().mean()
        loss = spatial + composition / self.fit_config["inventory_relative_scale"]**2 + regularization
        self.latest = {"coupled": coupled, "spatial_loss": float(spatial.detach()),
                       "inventory_relative_rms": float(torch.sqrt(composition.detach())),
                       "stage_inventory_relative_rms": {key: float(observed[key][3].detach().square().mean().sqrt())
                                                        for key in self.keys},
                       "stage_errors": {key: {name: float(value.detach().square().mean().sqrt())
                                            for name, value in observed[key][1].items()} for key in self.keys},
                       "stage_maximum_relative_errors": {
                           key: {name: float(value.detach().abs().max())
                                 for name, value in observed[key][1].items()} for key in self.keys},
                       "stage_scaled_rms": {key: {name: float(value.detach().square().mean().sqrt())
                                                for name, value in observed[key][2].items()} for key in self.keys}}
        if coupled:
            self.latest["final_cycle_metrics"] = {name: float(value.detach()) for name, value in metrics.items()}
        self.latest_predictions = predicted
        return loss

    def acceptable(self):
        """Check the latest field and inventory residual acceptance limits.

        Returns
        -------
        bool
            Whether every configured field, whole-history and stage inventory criterion passes.
        """
        worst = max(value for errors in self.latest["stage_scaled_rms"].values() for value in errors.values())
        return (worst <= self.fit_config["maximum_family_scaled_rms"]
                and self.latest["inventory_relative_rms"] <= self.fit_config["maximum_inventory_relative_rms"]
                and max(self.latest["stage_inventory_relative_rms"].values())
                    <= self.fit_config["maximum_stage_inventory_relative_rms"])


def fit_shared(model, *, coupled, progress):
    """Fit shared coefficients with bounded scalar-loss LBFGS work.

    Parameters
    ----------
    model : SharedCycle
        Calibration model carrying its fit configuration and coefficient parameters.
    coupled : bool
        Fit the connected depletion trajectory; False pools responses at fixed native
        inventories.
    progress : callable
        Progress callback receiving named status, timing or metric updates.

    Returns
    -------
    dict
        Optimization trace, selected coefficients, elapsed time, residual checks and
        optional connected predictions.
    """
    config = model.fit_config
    steps = config["coupled_iterations"] if coupled else config["pooled_iterations"]
    options = {"max_iter": steps, "max_eval": config["coupled_max_evaluations"] if coupled else 2 * steps,
               "history_size": 10, "tolerance_grad": 1e-6, "tolerance_change": 1e-8, "line_search_fn": "strong_wolfe"}
    optimizer = torch.optim.LBFGS(model.coefficients.parameters(), **options)
    trace = []
    started = time.perf_counter()

    def closure():
        """Evaluate the current calibration loss and its coefficient gradient.

        Returns
        -------
        torch.Tensor
            Finite scalar loss passed to the LBFGS line search.
        """
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        loss = model(coupled=coupled)
        forward_s = time.perf_counter() - tick
        if not torch.isfinite(loss):
            raise ArithmeticError("Nonfinite shared-cycle calibration loss")
        if coupled:
            progress(layout=model.layout, phase="coupled_backward", forward_wall_s=forward_s,
                     evaluations=len(trace) + 1, loss=float(loss.detach()))
        tick = time.perf_counter()
        loss.backward()
        backward_s = time.perf_counter() - tick
        norm = float(model.coefficients.raw.grad.abs().max())
        trace.append({"loss": float(loss.detach()), "gradient_inf": norm,
                      "forward_wall_s": forward_s, "backward_wall_s": backward_s})
        progress(layout=model.layout, phase="coupled" if coupled else "pooled",
                 parameters=len(model.parameter_names), evaluations=len(trace), loss=trace[-1]["loss"],
                 gradient_inf=norm, inventory_relative_rms=model.latest["inventory_relative_rms"])
        return loss
    if steps:
        optimizer.step(closure)
    else:
        progress(layout=model.layout, phase="coupled_validation" if coupled else "pooled_validation",
                 parameters=len(model.parameter_names), evaluations=0)
    with torch.no_grad():
        loss = model(coupled=coupled)
    result = {"trace": trace, "loss": float(loss), "log_factors": model.coefficients().detach().tolist(),
              "wall_s": time.perf_counter() - started,
              "optimizer_iterations": optimizer.state[model.coefficients.raw].get("n_iter", 0),
              "optimization_performed": bool(steps),
              "errors": model.latest, "accepted": model.acceptable()}
    if coupled:
        result["predictions"] = model.latest_predictions
    return result


def calibration_identity(root):
    """Identify the effective inputs and native evidence of a fit.

    Parameters
    ----------
    root : CycleCase
        Resolved case providing model factories, settings and result paths.

    Returns
    -------
    dict
        Readable case definition and calibration settings, with native-data checksums.
    """
    workflow = root
    return {"case_name": workflow.name,
            "calibration_settings": workflow.settings("calibration"),
            "native_history_sha256": sha(workflow.result("history")),
            "native_states_manifest_sha256": sha(workflow.result("states", "") / "manifest.json"),
            "case_definition": workflow.definition()}


def calibrate_shared(root, physical, chain, mapping, entry, history, *, progress):
    """Select a spatial basis and fit its connected depletion history.

    Parameters
    ----------
    root : CycleCase
        Resolved case providing model factories, settings and result paths.
    physical : IsotopicInventory
        Full-chain inventory binding, native case factory and live fresh-fuel recipe.
    chain : SparseChain
        Differentiable full-chain propagator accepting inventory, reaction frequencies in
        s⁻¹ and elapsed seconds.
    mapping : FuelMap
        Region-to-channel/axial mapping with physical multiplicities and symmetry fractions.
    entry : FuelState
        Region-ordered inventories, carried energy in J, initial heavy-metal masses in kg
        and physical fuel identities.
    history : mapping
        Retained native cycle schedule, observations, design and ordered stage keys.
    progress : callable
        Progress callback receiving named status, timing or metric updates.

    Returns
    -------
    dict
        Saved candidate fits and selected connected fit, including explicit field-quality
        status.

    Raises
    ------
    ArithmeticError
        The connected fit fails required criteria and exploratory fitting is disabled.
    """
    workflow = root
    config = workflow.settings("calibration")
    result = {"status": "calibrating", "scope": config["scope"], **calibration_identity(workflow),
              "candidate_bases": [], "selected": None, "selection_policy": config["selection_policy"]}
    path = workflow.result("fit")
    started = time.perf_counter()

    def save():
        """Atomically save calibration progress and elapsed time."""
        result["wall_s"] = time.perf_counter() - started
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)

    def build(layout):
        """Build a fresh calibration model at the native history design.

        Parameters
        ----------
        layout : str
            Case-declared spatial correction-basis layout.

        Returns
        -------
        SharedCycle
            Shared trajectory model for the requested basis layout.
        """
        design = workflow.design()
        with torch.no_grad():
            for name, value in history["design"].items():
                getattr(design, name).copy_(tensor(value))
        return SharedCycle(workflow, physical, chain, mapping, entry, history, design, layout)

    try:
        for layout in config["layouts"]:
            model = build(layout)
            initial = None
            for row in result["candidate_bases"]:
                case = model.models[0].case
                names = workflow.model.equivalence_bases(case, layout=row["layout"], include_void=False)[0]
                if list(names) != row["parameter_names"]:
                    continue
                values = transfer_coefficients(case, row["layout"], row["pooled"]["log_factors"],
                                               layout, config["log_factor_bound"],
                                               basis_factory=workflow.model.equivalence_bases)
                if values is None:
                    continue
                model.coefficients.initialize(values)
                with torch.no_grad():
                    loss = float(model(coupled=False))
                if initial is None or loss < initial["loss"]:
                    initial = {"layout": row["layout"], "loss": loss, "log_factors": values.tolist()}
            if initial is not None:
                model.coefficients.initialize(initial["log_factors"])
            progress(status="calibrating_shared_cycle", layout=layout, phase="pooled",
                     parameters=len(model.parameter_names), initialization=None if initial is None
                     else {key: initial[key] for key in ("layout", "loss")})
            pooled = fit_shared(model, coupled=False, progress=progress)
            result["candidate_bases"].append({"layout": layout,
                "parameter_names": list(model.parameter_names), "initialization": initial, "pooled": pooled})
            save()
            del model
        best = min(row["pooled"]["loss"] for row in result["candidate_bases"])
        candidates = [row for row in result["candidate_bases"]
                      if row["pooled"]["loss"] <= best * (1 + config["basis_selection_relative_loss_allowance"])]
        selected = min(candidates, key=lambda row: len(row["parameter_names"]))
        if config.get("selected_layout") is not None:
            selected = next(row for row in result["candidate_bases"] if row["layout"] == config["selected_layout"])
        model = build(selected["layout"])
        if list(model.parameter_names) != selected["parameter_names"]:
            raise ValueError("Selected pooled basis differs from the current spatial basis")
        model.coefficients.initialize(selected["pooled"]["log_factors"])
        result["selected"] = selected
        save()
        progress(status="calibrating_shared_cycle", phase="coupled", layout=model.layout,
                 parameters=len(model.parameter_names))
        selected["coupled"] = fit_shared(model, coupled=True, progress=progress)
        result["field_quality_passed"] = selected["coupled"]["accepted"]
        result["status"] = "checked" if result["field_quality_passed"] else "exploratory"
        save()
        if not result["field_quality_passed"] and not config["allow_exploratory_fit"]:
            raise ArithmeticError("Shared trajectory fit did not meet the field/inventory criteria")
        return result
    except BaseException as error:
        result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        save()
        raise


def restore_shared(root, physical, chain, mapping, entry, history, design):
    """Restore a matching fit and freeze its shared correction factors.

    Parameters
    ----------
    root : CycleCase
        Resolved case providing model factories, settings and result paths.
    physical : IsotopicInventory
        Full-chain inventory binding, native case factory and live fresh-fuel recipe.
    chain : SparseChain
        Differentiable full-chain propagator accepting inventory, reaction frequencies in
        s⁻¹ and elapsed seconds.
    mapping : FuelMap
        Region-to-channel/axial mapping with physical multiplicities and symmetry fractions.
    entry : FuelState
        Region-ordered inventories, carried energy in J, initial heavy-metal masses in kg
        and physical fuel identities.
    history : mapping
        Retained native cycle schedule, observations, design and ordered stage keys.
    design : torch.nn.Module
        Live case design, including the common cycle_days parameter.

    Returns
    -------
    model : SharedCycle
        Model with fixed calibration coefficients and live design parameters.
    record : dict
        Validated saved calibration record.

    Raises
    ------
    ValueError
        Status, evidence identity or spatial-basis names differ from the current case.
    """
    workflow = root
    result = json.loads(workflow.result("fit").read_text())
    config = workflow.settings("calibration")
    allowed = ("checked", "exploratory") if config["allow_exploratory_fit"] else ("checked",)
    if result["status"] not in allowed or any(result.get(name) != value
            for name, value in calibration_identity(workflow).items()):
        raise ValueError("Shared calibration is incomplete or differs from the current history/model")
    selected = result["selected"]
    model = SharedCycle(workflow, physical, chain, mapping, entry, history, design, selected["layout"])
    if list(model.parameter_names) != selected["parameter_names"]:
        raise ValueError("Shared calibration basis changed")
    model.coefficients.initialize(selected["coupled"]["log_factors"])
    model.coefficients.requires_grad_(False)
    design.requires_grad_(True)
    return model, result
