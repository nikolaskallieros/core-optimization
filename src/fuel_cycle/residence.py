"""Synchronized refuelling and a connected full-residence diffusion/depletion graph."""
from dataclasses import dataclass, replace

import torch
from optimization.objectives import temporal_power

from diffusion_surrogate.solver import tensor

DAY = 86400.
J_PER_MWD = 1e6 * DAY


@dataclass
class FuelState:
    """Zone densities in atoms/cm3, carried energy in J, initial HM in kg and fuel IDs."""
    inventory: torch.Tensor
    energy_j: torch.Tensor
    initial_hm_kg: torch.Tensor
    ids: tuple[str, ...]

    def detached(self):
        """Copy a fuel state without retaining its derivative graph.

        Returns
        -------
        FuelState
            Independent inventory, energy and mass tensors with unchanged fuel identities.
        """
        return FuelState(self.inventory.detach().clone(), self.energy_j.detach().clone(),
                         self.initial_hm_kg.detach().clone(), self.ids)


class FuelMap:
    """One tracked state per channel piece and axial assembly, with exact weights."""
    def __init__(self, physical):
        """Map ordered fuel regions to physical channels and axial levels.

        Parameters
        ----------
        physical : IsotopicInventory
            Full-chain inventory binding, native case factory and live fresh-fuel recipe.
        """
        self.physical = physical
        self.indices = {label: i for i, label in enumerate(physical.labels)}
        self.rings = torch.tensor([r["ring"] for r in physical.regions])
        self.levels = torch.tensor([int(r["label"].rpartition("/axial/")[2]) for r in physical.regions])
        self.channels = tuple(r["label"].rpartition("/axial/")[0] for r in physical.regions)
        self.count = int(self.levels.max()) + 1
        self.fractions = tensor(physical.multiplicity / physical.symmetry_multiplier)
        self.channel_ids = tuple(dict.fromkeys(self.channels))
        self.channel_indices = [torch.tensor([i for i, c in enumerate(self.channels) if c == channel])
                                for channel in self.channel_ids]
        self.channel_fractions = torch.stack([self.fractions[i[0]] for i in self.channel_indices])

    def shuffle(self, state, rings, fresh, fresh_hm, tag):
        """Move selected fuel stacks and retain the discharged histories.

        Parameters
        ----------
        state : FuelState
            Region-ordered inventories, carried energy in J, initial heavy-metal masses in kg
            and physical fuel identities.
        rings : collection of int
            Ring indices whose channels insert one fresh top assembly.
        fresh : torch.Tensor
            Fresh region-by-chain atom densities in atoms/cm³.
        fresh_hm : torch.Tensor
            Fresh initial heavy-metal mass in kg for each represented region.
        tag : str
            Insertion label used to create unique physical fuel identities.

        Returns
        -------
        moved : FuelState
            Updated stack inventories and histories, preserving their live gradient paths.
        outgoing : dict
            Discharged region indices, fuel identities, carried energy and initial masses.
        """
        selected = torch.tensor([int(r) in rings for r in self.rings])
        inserted = selected & (self.levels == 0)
        discharged = selected & (self.levels == self.count - 1)
        sources = [self.indices[f"{channel}/axial/{int(level) - 1}"] if moved and level > 0 else i
                   for i, (channel, level, moved) in enumerate(zip(self.channels, self.levels, selected))]
        ids = tuple(f"{channel}/{tag}" if bool(inserted[i]) else state.ids[sources[i]]
                    for i, channel in enumerate(self.channels))
        outgoing = {"indices": discharged.nonzero().flatten(),
                    "energy_j": state.energy_j[discharged], "initial_hm_kg": state.initial_hm_kg[discharged],
                    "ids": tuple(state.ids[i] for i in discharged.nonzero().flatten().tolist())}
        moved = replace(state, inventory=torch.where(inserted[:, None], fresh, state.inventory[sources]),
                        energy_j=torch.where(inserted, torch.zeros_like(state.energy_j), state.energy_j[sources]),
                        initial_hm_kg=torch.where(inserted, fresh_hm, state.initial_hm_kg[sources]), ids=ids)
        if len(set(moved.ids)) != len(moved.ids):
            raise ArithmeticError("Fuel identity duplicated during loading")
        return moved, outgoing

    def physical_power(self, power_w):
        """Reconstruct complete physical assembly powers from region powers.

        Parameters
        ----------
        power_w : torch.Tensor
            Represented fuel-region powers in W, including each region's symmetry fraction.

        Returns
        -------
        torch.Tensor
            Complete assembly power in W for each represented axial position.
        """
        return power_w / self.fractions

    def channel_power(self, power_w):
        """Sum complete axial assembly powers into radial channel powers.

        Parameters
        ----------
        power_w : torch.Tensor
            Represented fuel-region powers in W.

        Returns
        -------
        torch.Tensor
            Complete physical channel powers in W, in channel_ids order.
        """
        physical = self.physical_power(power_w)
        return torch.stack([physical[indices].sum() for indices in self.channel_indices])

    def burnup(self, state):
        """Convert carried energy to burnup per initial heavy-metal mass.

        Parameters
        ----------
        state : FuelState
            Region-ordered inventories, carried energy in J, initial heavy-metal masses in kg
            and physical fuel identities.

        Returns
        -------
        torch.Tensor
            Region-ordered burnup in MWd/tHM.
        """
        return state.energy_j / J_PER_MWD / (state.initial_hm_kg / 1000)


def schedule(cycle_days, step_fractions):
    """Partition a differentiable common refuelling period.

    Parameters
    ----------
    cycle_days : float or torch.Tensor
        Positive common refuelling period in days.
    step_fractions : sequence of float
        Strictly increasing cycle fractions starting at zero and ending at one.

    Returns
    -------
    list of tuple
        Pairs of live segment-end time in days and stable step identifiers.
    """
    cycle_days = tensor(cycle_days)
    if cycle_days.numel() != 1 or not torch.isfinite(cycle_days).all() or bool((cycle_days <= 0).any()):
        raise ValueError("Require one finite positive cycle duration")
    cycle_days = cycle_days.reshape(())
    if step_fractions[0] != 0 or step_fractions[-1] != 1 or any(b <= a for a, b in zip(step_fractions, step_fractions[1:])):
        raise ValueError("Step fractions must strictly partition the common cycle")
    return [(cycle_days * value, f"step/{i}") for i, value in enumerate(step_fractions[1:])]


def native_output(point):
    """Extract measured native rates, power and criticality.

    Parameters
    ----------
    point : NativePoint or ResponsePoint
        State carrying retained native transport observations.

    Returns
    -------
    dict
        Per-parent rates in s⁻¹, powers in W, keff, reference boron and group flux in
        n/cm²/s.
    """
    return {"rates": point.native_rates_by_group.sum(-1), "zone_power_w": point.native_power,
            "power_w": tensor(point.state["case"]["power_W"]), "keff": tensor(point.native_keff),
            "boron_ppm": tensor(point.reference_boron), "group_flux_cm2_s": point.native_flux}


def local_output(point, inventory, *, detach_inventory=False):
    """Evaluate a response at its retained reference boron concentration.

    Parameters
    ----------
    point : ResponsePoint
        Local or shared surrogate response to evaluate.
    inventory : torch.Tensor
        CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).
    detach_inventory : bool, optional
        Freeze inventory derivatives for a diagnostic evaluation.

    Returns
    -------
    dict
        Stage response with control_status set to fixed_boron.
    """
    if detach_inventory:
        inventory = inventory.detach()
    output = point.evaluate(inventory, tensor(point.reference_boron))
    output["control_status"] = "fixed_boron"
    return output


def cecm(state, duration_days, start, midpoint, chain):
    """Advance connected inventories and energy through one CE/CM step.

    Parameters
    ----------
    state : FuelState
        Region-ordered inventories, carried energy in J, initial heavy-metal masses in kg
        and physical fuel identities.
    duration_days : torch.Tensor
        Live interval duration in days.
    start : mapping
        Beginning-stage outputs containing per-parent reaction frequencies in s⁻¹.
    midpoint : callable
        Evaluate the predicted midpoint inventory and return rates and zone_power_w.
    chain : SparseChain
        Differentiable full-chain propagator accepting inventory, reaction frequencies in
        s⁻¹ and elapsed seconds.

    Returns
    -------
    final : FuelState
        Corrected endpoint inventory and midpoint-integrated energy.
    middle_inventory : torch.Tensor
        Predictor midpoint atom densities in atoms/cm³.
    middle : mapping
        Outputs evaluated at the actual predictor midpoint inventory.
    """
    seconds = duration_days * DAY
    middle_inventory = chain(state.inventory, start["rates"], seconds / 2)
    middle = midpoint(middle_inventory)
    final_inventory = chain(state.inventory, middle["rates"], seconds)
    energy = state.energy_j + middle["zone_power_w"] * seconds
    return replace(state, inventory=final_inventory, energy_j=energy), middle_inventory, middle


def metrics(mapping, state, samples, discharges, config):
    """Evaluate discharge burnup and sampled power/criticality objectives.

    Parameters
    ----------
    mapping : FuelMap
        Region-to-channel/axial mapping with physical multiplicities and symmetry fractions.
    state : FuelState
        Region-ordered inventories, carried energy in J, initial heavy-metal masses in kg
        and physical fuel identities.
    samples : sequence of mapping
        Segment durations in days, stage powers in W, endpoint keff and peak burnup.
    discharges : sequence of mapping
        Discharged represented-region energies in J and initial heavy-metal masses in kg.
    config : mapping
        Resolved cycle, refuelling, objective and sampling settings.

    Returns
    -------
    dict of str to torch.Tensor
        Scalar loss, discharge and peak burnup in MWd/tHM, dimensionless power variances and
        keff deficit, plus power swing in MW when selected.
    """
    if not discharges:
        raise ValueError("No discharged batches in the final cycle")
    energy = torch.cat([d["energy_j"] for d in discharges]).sum()
    mass = torch.cat([d["initial_hm_kg"] for d in discharges]).sum()
    burnup = energy / J_PER_MWD / (mass / 1000)
    durations = torch.stack([s["duration_days"] for s in samples])
    flat, channel_flat, deficits = [], [], []
    objective = config["objective"]
    for sample in samples:
        p = mapping.physical_power(sample["zone_power_w"])
        weights = mapping.fractions / mapping.fractions.sum()
        mean = (weights * p).sum()
        flat.append((weights * (p / mean - 1).square()).sum())
        channel_p = mapping.channel_power(sample["zone_power_w"])
        channel_weights = mapping.channel_fractions / mapping.channel_fractions.sum()
        channel_mean = (channel_weights * channel_p).sum()
        channel_flat.append((channel_weights * (channel_p / channel_mean - 1).square()).sum())
        deficits.append(torch.relu(objective.get("minimum_keff", 1.) - sample["keff"]).square())
    assembly_flatness = (torch.stack(flat) * durations).sum() / durations.sum()
    channel_flatness = (torch.stack(channel_flat) * durations).sum() / durations.sum()
    flatness = channel_flatness if objective.get("power_basis") == "channel" else assembly_flatness
    temporal = None
    if objective.get("power_basis") == "channel_temporal":
        temporal = sample_temporal_power(mapping, samples)
        flatness = temporal["variance"]
    deficit = (torch.stack(deficits) * durations).sum() / durations.sum()
    peak = torch.stack([s["peak_burnup"] for s in samples]).max()
    peak_limit = objective.get("peak_burnup_MWd_tHM")
    peak_penalty = torch.relu(peak / peak_limit - 1).square() if peak_limit is not None else 0.
    loss = (-burnup / objective["reference_discharge_burnup_MWd_tHM"]
            + objective.get("channel_power_weight", objective.get("assembly_power_weight", 1.)) * flatness
            + deficit / objective["keff_deficit_scale"]**2
            + peak_penalty)
    if temporal is not None:
        loss = (flatness / objective["temporal_variance_scale"]
                + (torch.relu(objective["minimum_discharge_burnup_MWd_tHM"] - burnup)
                   / objective["burnup_penalty_scale_MWd_tHM"]).square()
                + deficit / objective["keff_deficit_scale"]**2)
    temporal_metrics = ({"maximum_channel_power_swing_MW": temporal["maximum_power_swing_W"] / 1e6}
                        if temporal is not None else {})
    return {"loss": loss, "discharge_burnup_MWd_tHM": burnup, **temporal_metrics,
            "power_variance": flatness, "rms_keff_deficit": torch.sqrt(deficit),
            "channel_power_variance": channel_flatness, "assembly_power_variance": assembly_flatness,
            "peak_burnup_MWd_tHM": peak}


def replay_cycle(mapping, chain, design, pre_cycle, record, points, config, *, detach_inventory=False,
                  gradient_trace=None, fixed_cycle_days=None, observe=None, entered=False):
    """Advance one cycle while preserving every live inventory update.

    Parameters
    ----------
    mapping : FuelMap
        Region-to-channel/axial mapping with physical multiplicities and symmetry fractions.
    chain : SparseChain
        Differentiable full-chain propagator accepting inventory, reaction frequencies in
        s⁻¹ and elapsed seconds.
    design : torch.nn.Module
        Live case design, including the common cycle_days parameter.
    pre_cycle : FuelState
        Region-ordered inventories, carried energy in J, initial heavy-metal masses in kg
        and physical fuel identities.
    record : mapping
        Native cycle record containing BOL and segment start/midpoint/end keys.
    points : mapping of str to ResponsePoint
        State-keyed native or surrogate responses for every declared cycle stage.
    config : mapping
        Resolved cycle, refuelling, objective and sampling settings.
    detach_inventory : bool, optional
        Freeze inventory dependence only when evaluating stage responses.
    gradient_trace : list or None, optional
        Optional list receiving backward-stage inventory sensitivity summaries.
    fixed_cycle_days : float or None, optional
        Optional fixed period in days; None retains derivatives through design.cycle_days.
    observe : callable or None, optional
        Optional observe(key, inventory, output) callback for each evaluated stage.
    entered : bool, optional
        Use the supplied state directly; otherwise detach and rescale it before the initial
        shuffle.

    Returns
    -------
    objectives : dict
        Cycle objective tensors.
    final : FuelState
        Resident state before the next cycle's insertion.
    samples : list of dict
        Evaluated segment durations, powers, criticality and peak burnup.
    cycle_days : torch.Tensor
        Common period in days, retaining its design derivative unless fixed explicitly.
    """
    fresh, volumes, hm = mapping.physical.live_recipe(design)
    if entered:
        state = pre_cycle
    else:
        scale = volumes / tensor(mapping.physical.volumes)
        state = pre_cycle.detached()
        state = replace(state, energy_j=state.energy_j * scale, initial_hm_kg=state.initial_hm_kg * scale)
        state, _ = mapping.shuffle(state, config["refueling"]["rings"], fresh, hm, f"cycle{record['cycle']}/BOL")
    first = local_output(points[record["bol"]], state.inventory, detach_inventory=detach_inventory)
    if observe is not None:
        observe(record["bol"], state.inventory, first)
    cycle_days = design.cycle_days.reshape(()) if fixed_cycle_days is None else tensor(fixed_cycle_days)
    events = schedule(cycle_days, config["step_fractions"])
    if [e[1] for e in events] != [s["end_key"] for s in record["segments"]] or any(
            s["reload_rings"] for s in record["segments"]):
        raise ValueError("Native history does not use the current common-cycle schedule")
    samples, discharges = [], []
    time = tensor(0.)
    for segment, (end_time, _) in zip(record["segments"], events):
        start = local_output(points[segment["start"]], state.inventory, detach_inventory=detach_inventory)
        if observe is not None:
            observe(segment["start"], state.inventory, start)
        dt = end_time - time

        def midpoint(inventory):
            """Evaluate and optionally observe the predictor midpoint.

            Parameters
            ----------
            inventory : torch.Tensor
                CPU float64 fuel atom densities in atoms/cm³, shaped (fuel_regions, chain_nuclides).

            Returns
            -------
            dict
                Midpoint response evaluated on the propagated inventory.
            """
            output = local_output(points[segment["mid"]], inventory, detach_inventory=detach_inventory)
            if observe is not None:
                observe(segment["mid"], inventory, output)
            return output

        state, middle_inventory, middle = cecm(
            state, dt, start, midpoint, chain)
        if gradient_trace is not None and state.inventory.requires_grad:
            key, density = segment["end"], state.inventory.detach()
            def observe_gradient(gradient, key=key, density=density):
                """Record one inventory's logarithmic sensitivity norm during backward.

                Parameters
                ----------
                gradient : torch.Tensor
                    Incoming inventory cotangent.
                key : str, optional
                    Captured endpoint identifier.
                density : torch.Tensor, optional
                    Captured endpoint densities in atoms/cm³, detached from the forward graph.
                """
                gradient_trace.append({"stage": key, "inventory_log_sensitivity_norm":
                                       float(torch.linalg.vector_norm(gradient * density))})
            state.inventory.register_hook(observe_gradient)
        end = local_output(points[segment["end"]], state.inventory, detach_inventory=detach_inventory)
        if observe is not None:
            observe(segment["end"], state.inventory, end)
        samples.append({"duration_days": dt, "time_days": end_time, "zone_power_w": middle["zone_power_w"],
                        "start_zone_power_w": start["zone_power_w"], "end_zone_power_w": end["zone_power_w"],
                        "keff": end["keff"], "boron_ppm": end["boron_ppm"],
                        "peak_burnup": mapping.burnup(state).max()})
        time = end_time
    # Count all discharged fuel. The next insertion belongs to the next cycle.
    _, outgoing = mapping.shuffle(state, config["refueling"]["rings"], fresh, hm, f"cycle{record['cycle'] + 1}/BOL")
    discharges.append(outgoing)
    return metrics(mapping, state, samples, discharges, config), state, samples, cycle_days


def replay_history(mapping, chain, design, history, points, config, *, observe=None,
                   gradient_trace=None, fixed_cycle_days=None):
    """Replay the complete residence with shared differentiable geometry.

    Parameters
    ----------
    mapping : FuelMap
        Region-to-channel/axial mapping with physical multiplicities and symmetry fractions.
    chain : SparseChain
        Differentiable full-chain propagator accepting inventory, reaction frequencies in
        s⁻¹ and elapsed seconds.
    design : torch.nn.Module
        Live case design, including the common cycle_days parameter.
    history : mapping
        Retained native cycle schedule, observations, design and ordered stage keys.
    points : mapping of str to ResponsePoint
        State-keyed native or surrogate responses for every declared cycle stage.
    config : mapping
        Resolved cycle, refuelling, objective and sampling settings.
    observe : callable or None, optional
        Optional observe(key, inventory, output) callback for each evaluated stage.
    gradient_trace : list or None, optional
        Optional list receiving backward-stage inventory sensitivity summaries.
    fixed_cycle_days : float or None, optional
        Optional fixed period in days; None retains derivatives through design.cycle_days.

    Returns
    -------
    objectives : dict
        Final-cycle objective tensors.
    final : FuelState
        Final resident state.
    cycles : list of dict
        Per-cycle metrics, endpoint states and sampled stages.
    cycle_days : torch.Tensor
        Common refuelling period in days.
    """
    from diffusion_surrogate.regional import shared_geometry
    with shared_geometry(points):
        return _replay_history(mapping, chain, design, history, points, config, observe=observe,
                               gradient_trace=gradient_trace, fixed_cycle_days=fixed_cycle_days)


def _replay_history(mapping, chain, design, history, points, config, *, observe=None,
                    gradient_trace=None, fixed_cycle_days=None):
    """Connect fresh loading, all refuelling events and final discharge.

    Parameters
    ----------
    mapping : FuelMap
        Region-to-channel/axial mapping with physical multiplicities and symmetry fractions.
    chain : SparseChain
        Differentiable full-chain propagator accepting inventory, reaction frequencies in
        s⁻¹ and elapsed seconds.
    design : torch.nn.Module
        Live case design, including the common cycle_days parameter.
    history : mapping
        Retained native cycle schedule, observations, design and ordered stage keys.
    points : mapping of str to ResponsePoint
        State-keyed native or surrogate responses for every declared cycle stage.
    config : mapping
        Resolved cycle, refuelling, objective and sampling settings.
    observe : callable or None, optional
        Optional observe(key, inventory, output) callback for each evaluated stage.
    gradient_trace : list or None, optional
        Optional list receiving backward-stage inventory sensitivity summaries.
    fixed_cycle_days : float or None, optional
        Optional fixed period in days; None retains derivatives through design.cycle_days.

    Returns
    -------
    objectives : dict
        Final-cycle objective tensors.
    final : FuelState
        Final resident state.
    cycles : list of dict
        Per-cycle metrics, endpoint states and sampled stages.
    cycle_days : torch.Tensor
        Common refuelling period in days.
    """
    if config["initial_inventory"] != "fresh" or config["objective_cycle"] != len(history["cycles"]):
        raise ValueError("The full-residence test requires fresh loading and a final-cycle objective")
    if config["differentiated_cycles"] != [record["cycle"] for record in history["cycles"]]:
        raise ValueError("Every simulated cycle must participate in the derivative")
    fresh, _, hm = mapping.physical.live_recipe(design)
    state = FuelState(fresh, fresh.new_zeros(len(fresh)), hm,
                      tuple(f"cycle1/{label}" for label in mapping.physical.labels))
    cycle_results = []
    for record in history["cycles"]:
        if record["cycle"] > 1:
            state, _ = mapping.shuffle(state, config["refueling"]["rings"], fresh, hm,
                                       f"cycle{record['cycle']}/BOL")
        result, state, samples, days = replay_cycle(
            mapping, chain, design, state, record, points, config, entered=True,
            observe=observe, gradient_trace=gradient_trace, fixed_cycle_days=fixed_cycle_days)
        cycle_results.append({"cycle": record["cycle"], "metrics": result,
                              "end_state": state, "samples": samples})
    return result, state, cycle_results, days


def state_payload(state):
    """Detach fuel-state arrays for a numerical record.

    Parameters
    ----------
    state : FuelState
        Region-ordered inventories, carried energy in J, initial heavy-metal masses in kg
        and physical fuel identities.

    Returns
    -------
    dict
        Inventory in atoms/cm³, energy in J, initial heavy-metal mass in kg and unchanged
        fuel IDs.
    """
    return {"inventory": state.inventory.detach().numpy(), "energy_j": state.energy_j.detach().numpy(),
            "initial_hm_kg": state.initial_hm_kg.detach().numpy(), "ids": state.ids}


def restore_state(record):
    """Restore ordered fuel-state tensors from a numerical record.

    Parameters
    ----------
    record : mapping
        Saved inventory, energy_j, initial_hm_kg and ids fields.

    Returns
    -------
    FuelState
        CPU float64 state using the saved region and chain ordering.
    """
    return FuelState(tensor(record["inventory"]), tensor(record["energy_j"]), tensor(record["initial_hm_kg"]),
                     tuple(record["ids"]))


def sample_temporal_power(mapping, samples):
    """Integrate channel power through all sampled cycle stages.

    Parameters
    ----------
    mapping : FuelMap
        Region-to-channel/axial mapping with physical multiplicities and symmetry fractions.
    samples : sequence of mapping
        Segments containing beginning, midpoint and endpoint region powers in W and times in
        days.

    Returns
    -------
    dict
        Piecewise-linear physical-channel power means, variances, ranges and limiting times.
    """
    times = [samples[0]["time_days"] - samples[0]["duration_days"]]
    powers = [mapping.channel_power(samples[0]["start_zone_power_w"])]
    for sample in samples:
        times.extend((sample["time_days"] - sample["duration_days"] / 2, sample["time_days"]))
        powers.extend((mapping.channel_power(sample["zone_power_w"]),
                       mapping.channel_power(sample["end_zone_power_w"])))
    return temporal_power(torch.stack(times), torch.stack(powers), mapping.channel_fractions)


def native_temporal_profile(history, mapping):
    """Recompute final-cycle power statistics from native observations.

    Parameters
    ----------
    history : mapping
        Retained native cycle schedule, observations, design and ordered stage keys.
    mapping : FuelMap
        Region-to-channel/axial mapping with physical multiplicities and symmetry fractions.

    Returns
    -------
    dict
        Serializable physical-channel power trajectory in W, sample times in days, temporal
        statistics and their definitions.
    """
    cycle = history["cycles"][-1]
    keys, times = [cycle["bol"]], [0.]
    for segment in cycle["segments"]:
        keys.extend((segment["mid"], segment["end"]))
        times.extend((segment["time_days"] - segment["duration_days"] / 2, segment["time_days"]))
    power = torch.stack([mapping.channel_power(tensor(history["observations"][key]["power_W"]))
                         for key in keys])
    values = temporal_power(times, power, mapping.channel_fractions)
    return {"sample_keys": keys, "time_days": times,
            "channel_ids": list(mapping.channel_ids),
            "channel_power_W": power.tolist(),
            **{name: value.tolist() for name, value in values.items()},
            "limiting_channel": mapping.channel_ids[int(values["limiting_channel_index"])],
            "power_swing_definition": "Maximum over radial positions of (maximum minus minimum total "
                                      "channel power over all sampled final-cycle states). Sum the axial "
                                      "assemblies before taking the range. Absolute watts, without multiplicity "
                                      "or mean-power weighting; no between-position power difference.",
            "definition": "Physical-channel-weighted temporal variance about each channel's own cycle mean. "
                          "Integrate piecewise-linear power through BOL, all CE/CM midpoints and endpoints. "
                          "Normalize each variance by its own squared mean power. No between-channel penalty."}
