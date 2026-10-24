"""Native histories, shared calibration and full-residence derivatives."""

from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from runtime.io import sha
from runtime.processes import interrupt_on_sigterm
from diffusion_surrogate.solver import tensor
from fuel_cycle.residence import (
    FuelState,
    FuelMap,
    DAY,
    J_PER_MWD,
    schedule,
    native_output,
    metrics,
    state_payload,
)
from depletion.sparse import SparseChain
from coupling.native import evaluate_native
from runtime.io import save_state


DENSITY_FLOOR = 1e12


class NativePoint:
    """Retained OpenMC values; the native history has no diffusion dependency."""

    def __init__(self, state, chain):
        """Validate retained native rates, flux and fission-power accounting.

        Parameters
        ----------
        state : mapping
            Retained case, transport results, full-chain inventory and nuclide ordering.
        chain : openmc.deplete.Chain
            Evaluated chain supplying fission energy release for each transported parent.

        Raises
        ------
        ArithmeticError
            Rates are invalid, fuel flux/power is nonpositive or fission heat disagrees with
            tallied power by more than one percent.
        """
        self.state = state
        case, result = state["case"], state["result"]
        self.nuclides = tuple(state["nuclides"])
        self.reference_inventory = tensor(state["fuel_inventory_atoms_cm3"])
        self.reference_boron = case.get("boron_mg_B10_per_kg", 0.)
        self.fuel_indices = [i for i, region in enumerate(case["regions"]) if region["depletable"]]
        self.native_keff = result["combined_keff"][0]
        self.native_rates_by_group = tensor(result["depletion_response"]["per_parent_s_by_group"])
        self.native_power = tensor(result["observables"]["region_power_W"]["mean"])[
            self.fuel_indices
        ]
        self.native_flux = tensor(result["depletion_response"]["mean_group_flux_cm2_s"])
        if (
            not torch.isfinite(self.native_rates_by_group).all()
            or (self.native_rates_by_group < 0).any()
        ):
            raise ArithmeticError("Invalid native depletion rates")
        if (self.native_power <= 0).any() or (self.native_flux <= 0).any():
            raise ArithmeticError("Every fuel state needs positive native power and group flux")
        parents = [self.nuclides.index(name) for name in case["depletion_nuclides"]]
        fission = case["depletion_reactions"].index("fission")
        q = (
            tensor(
                [
                    sum(
                        r.Q
                        for r in chain.nuclides[chain.nuclide_dict[name]].reactions
                        if r.type == "fission"
                    )
                    for name in case["depletion_nuclides"]
                ]
            )
            * 1.602176634e-19
        )
        volumes = tensor([case["regions"][i]["volume_cm3"] for i in self.fuel_indices])
        heat = (
            self.reference_inventory[:, parents]
            * self.native_rates_by_group[:, :, fission].sum(-1)
            * q
        ).sum(-1) * volumes
        self.fission_power_relative_error = float(
            ((heat - self.native_power) / self.native_power).abs().max()
        )
        if (
            not np.isfinite(self.fission_power_relative_error)
            or self.fission_power_relative_error > 0.01
        ):
            raise ArithmeticError(
                "Native fission-rate energy differs from its power tally by more than 1%"
            )

    def evaluate(self, inventory, boron):
        """Return fixed native values for numerical history replay.

        Parameters
        ----------
        inventory : torch.Tensor
            Region-by-chain CPU float64 atom densities in atoms/cm³.
        boron : float or torch.Tensor
            Boron-10 concentration in mg per kg of unborated moderator.

        Returns
        -------
        dict
            Retained rates, powers, keff, boron and fuel-group flux.

        Notes
        -----
        Inventory and boron arguments are accepted to match the response interface. They do not
        alter these retained native observations; this adapter is only for replay verification.
        """
        return native_output(self)


class NativePoints:
    """Evaluate and retain ordered OpenMC states for a native depletion history."""
    def __init__(self, physical, directory, config, progress):
        """Prepare ordered native-state storage and progress accounting.

        Parameters
        ----------
        physical : IsotopicInventory
            Full-chain inventory binding and native case factory.
        directory : str or pathlib.Path
            Directory containing or receiving the native state files.
        config : mapping
            Resolved native cycle, refuelling, sampling and initialization settings.
        progress : callable
            Callback receiving keyword progress, timing and metric updates.
        """
        self.physical, self.directory, self.config, self.progress = (
            physical,
            Path(directory),
            config,
            progress,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timings = {"native_s": 0.0, "depletion_s": 0.0}
        self.observations = {}

    def point(self, inventory, key):
        """Evaluate or reuse one native inventory and retain diagnostics.

        Parameters
        ----------
        inventory : torch.Tensor
            Region-by-chain CPU float64 atom densities in atoms/cm³.
        key : str
            Stable native state key, including its cycle and stage where applicable.

        Returns
        -------
        NativePoint
            Validated native response with observations recorded under the supplied key.
        """
        self.progress(stage=key)
        path = self.directory / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        state = evaluate_native(
            self.physical,
            inventory.detach().numpy(),
            self.config["initial_boron_mg_B10_per_kg"],
            self.config["sampling"],
            path,
            self.directory / "native",
            progress=self.progress,
        )
        self.timings["native_s"] += state["result"]["wall_s"]
        point = NativePoint(state, self.physical.chain)
        self.observations[key] = {
            "keff": state["result"]["combined_keff"],
            "power_W": point.native_power.tolist(),
            "group_flux_cm2_s": point.native_flux.tolist(),
            "native_wall_s": state["result"]["wall_s"],
            "fission_power_relative_error": point.fission_power_relative_error,
        }
        fuel = point.fuel_indices
        power_tally = state["result"]["observables"]["region_power_W"]
        se = np.asarray(power_tally["scale"])[fuel] / np.asarray(power_tally["mean"])[fuel]
        self.observations[key]["power_relative_se_rms"] = float(np.sqrt(np.mean(se**2)))
        self.observations[key]["power_relative_se_max"] = float(se.max())
        self.observations[key]["physical_inventory_checks"] = state["result"].get(
            "physical_inventory_checks"
        )
        self.progress(native_keff=state["result"]["combined_keff"], timing=self.timings)
        return point


def compare_cycles(previous, current, observations, *, inventory_floor=1e14):
    """Compare native fuel states and observations at equal cycle phases.

    Parameters
    ----------
    previous : mapping
        Earlier cycle record with cycle_days, end_state and sampled stage keys.
    current : mapping
        Later cycle record using the same period and sampling phases.
    observations : mapping
        Native state-keyed power and keff observations for both cycles.
    inventory_floor : float, optional
        Atom-density floor in atoms/cm³ for relative inventory comparisons.

    Returns
    -------
    dict
        Relative inventory, burnup and assembly-power differences and absolute keff changes;
        these include Monte Carlo noise.

    Raises
    ------
    ValueError
        Cycle durations or sampled phase counts differ.
    """
    if previous["cycle_days"] != current["cycle_days"]:
        raise ValueError("Equilibrium comparison requires the same refuelling period")
    before, after = previous["end_state"], current["end_state"]
    a, b = np.asarray(before["inventory"]), np.asarray(after["inventory"])
    scale = np.maximum(abs(a), abs(b))
    mask = scale >= inventory_floor
    relative = ((b - a) / np.maximum(scale, inventory_floor))[mask]
    burnup = (
        lambda s: np.asarray(s["energy_j"]) / J_PER_MWD / (np.asarray(s["initial_hm_kg"]) / 1000)
    )
    ba, bb = burnup(before), burnup(after)
    keys = lambda c: [c["bol"], *[s[part] for s in c["segments"] for part in ("mid", "end")]]
    first, second = keys(previous), keys(current)
    if len(first) != len(second):
        raise ValueError("Equilibrium comparison requires matching cycle phases")
    power, keff = [], []
    for old, new in zip(first, second):
        p, q = np.asarray(observations[old]["power_W"]), np.asarray(observations[new]["power_W"])
        power.extend(((q - p) / np.maximum(abs(p), 1.0)).tolist())
        keff.append(observations[new]["keff"][0] - observations[old]["keff"][0])
    return {
        "previous_cycle": previous["cycle"],
        "cycle": current["cycle"],
        "inventory_relative_rms": float(np.sqrt(np.mean(relative**2))) if relative.size else 0.0,
        "inventory_max_relative": float(np.max(abs(relative))) if relative.size else 0.0,
        "inventory_floor_atoms_cm3": inventory_floor,
        "burnup_relative_rms": float(np.sqrt(np.mean(((bb - ba) / np.maximum(abs(ba), 1.0)) ** 2))),
        "assembly_power_history_relative_rms": float(np.sqrt(np.mean(np.square(power)))),
        "keff_history_max_absolute_change": float(np.max(abs(np.asarray(keff)))),
        "scope": "Position-resolved EOL inventory/burnup and equal-phase power/keff repeatability; includes Monte Carlo noise. No convergence claim from cycle count alone.",
    }


def native_history(root, directory, config, progress, *, design_values=None, initialize=True):
    """Generate a full native CE/CM history with synchronized refuelling.

    Parameters
    ----------
    root : CycleCase
        Resolved case with model factories, input settings and result paths.
    directory : str or pathlib.Path
        Directory containing or receiving the native state files.
    config : mapping
        Resolved native cycle, refuelling, sampling and initialization settings.
    progress : callable
        Callback receiving keyword progress, timing and metric updates.
    design_values : mapping or None, optional
        Explicit design parameter values; None uses the cycle configuration.
    initialize : bool, optional
        Allow the case's declared initialization policy to adjust an infeasible fresh
        loading.

    Returns
    -------
    physical : IsotopicInventory
        Inventory and native-data binding for the retained design.
    chain : SparseChain
        Compiled full-chain propagator.
    mapping : FuelMap
        Fuel-region and physical-channel mapping.
    entry : FuelState
        Initial resident fuel state.
    history : dict
        Complete native observations and cycle records.

    Notes
    -----
    Native beginning and midpoint reaction rates drive every depletion segment. The function
    writes restart states and cycle history as work completes. It can launch many OpenMC
    evaluations; diffusion predictions do not generate this evidence.
    """
    workflow = root
    directory = Path(directory)
    refuel_rings = tuple(config["refueling"]["rings"])
    values = dict(config["design"] if design_values is None else design_values)
    cycle_days = tensor(values["cycle_days"]).reshape(())
    bounds = config["refueling"]["cycle_days_bounds"]
    if not bounds[0] <= float(cycle_days) <= bounds[1]:
        raise ValueError("Common refuelling period exceeds the declared search bounds")
    initialization, compiled = [], None
    with torch.no_grad():
        for attempt in range(config["initialization"]["maximum_attempts"] if initialize else 1):
            physical = workflow.inventory(values)
            mapping = FuelMap(physical)
            if set(refuel_rings) != set(mapping.rings.tolist()):
                raise ValueError("The common cycle must refuel every fuelled ring")
            store = NativePoints(physical, directory, config, progress)
            fresh_point = store.point(tensor(physical.fresh), "fresh")
            if compiled is None:
                compiled = SparseChain(
                    physical.chain,
                    fresh_point.state["case"]["depletion_nuclides"],
                    physical.reactions,
                )
            if config["initial_inventory"] != "fresh":
                raise ValueError("The full-residence test starts from fresh fuel")
            state = FuelState(
                tensor(physical.fresh),
                tensor(np.zeros(len(physical.labels))),
                tensor(physical.hm_kg),
                tuple(f"cycle1/{label}" for label in physical.labels),
            )
            beginning = fresh_point
            k = beginning.native_keff
            initialization.append(
                {
                    "design": values,
                    "initial_keff": beginning.state["result"]["combined_keff"],
                    "cycle_days": float(cycle_days),
                }
            )
            if k >= config["initialization"]["bol_target_keff"]:
                break
            if not initialize:
                if k >= 1.0:
                    break
                raise ValueError(f"Proposed design's starting loading is subcritical: k={k:.5f}")
            if workflow.model.initialize_loading is None:
                raise ValueError("The initial loading is subcritical and no initialization policy was supplied")
            values = workflow.model.initialize_loading(values, k, config["initialization"])
            progress(
                stage="initial_loading_feasibility",
                initialization=initialization,
                next_design=values,
            )
        else:
            raise ValueError("Initial loading feasibility search exhausted its declared attempts")

        design = workflow.design()
        for name, value in values.items():
            getattr(design, name).copy_(tensor(value))
        fresh, _, fresh_hm = physical.live_recipe(design)
        records, pre_cycle = [], state.detached()
        save_state(directory / "initial_state.json", state_payload(pre_cycle))
        start_key, point = "fresh", beginning
        for cycle in range(1, config["cycles"] + 1):
            progress(status="native_history", cycle=cycle)
            if cycle > 1:
                state, _ = mapping.shuffle(
                    state, refuel_rings, fresh, fresh_hm, f"cycle{cycle}/BOL"
                )
                start_key = f"cycle{cycle}/bol"
                point = store.point(state.inventory, start_key)
            if point.native_keff < 1.0:
                raise ValueError(
                    f"Cycle {cycle} BOL is subcritical at the declared fixed boron: k={point.native_keff:.5f}"
                )
            events = schedule(cycle_days, config["step_fractions"])
            record = {
                "cycle": cycle,
                "bol": start_key,
                "cycle_days": float(cycle_days),
                "segments": [],
                "discharges": [],
            }
            time_days = tensor(0.0)
            samples, discharges = [], []
            initial_energy = state.energy_j.sum()
            for i, (end_time, end_key) in enumerate(events):
                mid_key, endpoint_key = f"cycle{cycle}/s{i}_mid", f"cycle{cycle}/s{i}_end"
                duration = end_time - time_days
                tick = time.perf_counter()
                half = compiled(state.inventory, native_output(point)["rates"], duration * DAY / 2)
                store.timings["depletion_s"] += time.perf_counter() - tick
                middle = store.point(half, mid_key)
                tick = time.perf_counter()
                final = compiled(state.inventory, native_output(middle)["rates"], duration * DAY)
                store.timings["depletion_s"] += time.perf_counter() - tick
                state = replace(
                    state,
                    inventory=final,
                    energy_j=state.energy_j + middle.native_power * duration * DAY,
                )
                endpoint = store.point(state.inventory, endpoint_key)
                sample = {
                    "duration_days": duration,
                    "time_days": end_time,
                    "zone_power_w": middle.native_power,
                    "keff": tensor(endpoint.native_keff),
                    "start_zone_power_w": point.native_power,
                    "end_zone_power_w": endpoint.native_power,
                    "peak_burnup": mapping.burnup(state).max(),
                }
                samples.append(sample)
                record["segments"].append(
                    {
                        "start": start_key,
                        "mid": mid_key,
                        "end": endpoint_key,
                        "end_key": end_key,
                        "time_days": float(end_time),
                        "duration_days": float(duration),
                        "reload_rings": [],
                        "burnup_MWd_tHM": mapping.burnup(state).tolist(),
                        "fuel_ids_before_loading": state.ids,
                    }
                )
                start_key, point = endpoint_key, endpoint
                time_days = end_time
                save_state(
                    directory / "current_state.json",
                    {
                        **state_payload(state),
                        "cycle": cycle,
                        "segment_completed": i,
                        "design": values,
                        "timing": store.timings,
                    },
                )
                progress(
                    cycle=cycle,
                    segment=i + 1,
                    segments=len(events),
                    cycle_days=float(cycle_days),
                    elapsed_cycle_days=float(time_days),
                )
            generated = physical.base_case["power_W"] * cycle_days * DAY
            balance = (state.energy_j.sum() - initial_energy - generated) / generated
            if abs(float(balance)) > 1e-10:
                raise ArithmeticError("Cumulative energy was lost in depletion or a ring transfer")
            _, outgoing = mapping.shuffle(
                state, refuel_rings, fresh, fresh_hm, f"cycle{cycle + 1}/BOL"
            )
            discharges.append(outgoing)
            record["discharges"].append(
                {
                    "time_days": float(cycle_days),
                    "rings": list(refuel_rings),
                    "ids": outgoing["ids"],
                    "burnup_MWd_tHM": (
                        outgoing["energy_j"] / J_PER_MWD / (outgoing["initial_hm_kg"] / 1000)
                    ).tolist(),
                }
            )
            record.update(
                metrics={
                    k: float(v)
                    for k, v in metrics(mapping, state, samples, discharges, config).items()
                },
                energy_balance_relative=float(balance),
                end_state=state_payload(state),
            )
            if records:
                record["repeatability_to_previous"] = compare_cycles(
                    records[-1], record, store.observations
                )
            records.append(record)
            save_state(
                directory / "history.json",
                {
                    "design": values,
                    "cycles": records,
                    "initialization": initialization,
                    "timing": store.timings,
                    "observations": store.observations,
                },
            )
            progress(
                cycle_completed=cycle,
                metrics=record["metrics"],
                repeatability=record.get("repeatability_to_previous"),
            )
            gc.collect()
        return (
            physical,
            compiled,
            mapping,
            pre_cycle,
            {
                "design": values,
                "cycles": records,
                "initialization": initialization,
                "timing": store.timings,
                "observations": store.observations,
            },
        )


def cycle_point_keys(history):
    """Collect unique native stage keys in stable sorted order.

    Parameters
    ----------
    history : mapping
        Retained cycle schedules, native observations, design values and endpoint fuel
        states.

    Returns
    -------
    list of str
        All BOL, segment-start, midpoint and endpoint identifiers used by the history.
    """
    keys = set()
    for record in history["cycles"]:
        keys.add(record["bol"])
        for segment in record["segments"]:
            keys.update(segment[name] for name in ("start", "mid", "end"))
    return sorted(keys)


def retain_points(root, directory, history):
    """Retain native states and verified arrays for the completed history.

    Parameters
    ----------
    root : CycleCase
        Resolved case with model factories, input settings and result paths.
    directory : str or pathlib.Path
        Directory containing or receiving the native state files.
    history : mapping
        Retained cycle schedules, native observations, design values and endpoint fuel
        states.

    Returns
    -------
    pathlib.Path
        Destination containing state pairs and a manifest identifying current inputs and
        model sources.
    """
    from runtime.io import load_state

    workflow = root
    destination = workflow.result("states", "")
    files = []
    for key in cycle_point_keys(history):
        source = Path(directory) / f"{key}.json"
        load_state(source)  # Verify its array hash before retaining the pair.
        metadata = json.loads(source.read_text())
        for path in (source, source.with_name(metadata["array_file"])):
            target = destination / path.relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if sha(path) != sha(target):
                raise OSError("A retained native state differs from its source")
            files.append(
                {
                    "path": str(target.relative_to(destination)),
                    "sha256": sha(target),
                    "bytes": target.stat().st_size,
                }
            )
    requested = workflow.settings()["cycles"]
    manifest = {
        "cycle": history["cycles"][-1]["cycle"],
        "design": history["design"],
        "status": "complete" if len(history["cycles"]) == requested else "partial",
        "requested_cycles": requested,
        "case_definition": workflow.definition(),
        "history_sha256": sha(workflow.result("history")),
        "entry_sha256": sha(workflow.result("entry")),
        "state_keys": cycle_point_keys(history),
        "files": files,
        "cycles": [c["cycle"] for c in history["cycles"]],
        "scope": "Compact native calibration states for every time in the completed cycles; no solver dumps. Only listed files are current; earlier per-state fit files are unused.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def restore_baseline(root):
    """Restore a complete native baseline after checking its identities.

    Parameters
    ----------
    root : CycleCase
        Resolved case with model factories, input settings and result paths.

    Returns
    -------
    physical : IsotopicInventory
        Inventory and native-data binding for the retained design.
    chain : SparseChain
        Compiled full-chain propagator.
    mapping : FuelMap
        Fuel-region and physical-channel mapping.
    entry : FuelState
        Initial resident fuel state.
    history : dict
        Complete native observations and cycle records.

    Raises
    ------
    ValueError
        Input definitions, retained files or the required cycle count differ.
    """
    from fuel_cycle.residence import restore_state
    from runtime.io import load_state

    workflow = root
    directory = workflow.result("states", "")
    manifest = json.loads((directory / "manifest.json").read_text())
    source = workflow.native_states()
    if manifest.get("case_definition") != workflow.definition():
        raise ValueError("The native baseline uses different inputs; choose a new output directory")
    for name, digest in (
        ("history_sha256", sha(workflow.result("history"))),
        ("entry_sha256", sha(workflow.result("entry"))),
    ):
        if manifest[name] != digest:
            raise ValueError(f"Retained baseline changed: {name}")
    for file in manifest["files"]:
        if sha(source / file["path"]) != file["sha256"]:
            raise ValueError(f"Retained calibration state changed: {file['path']}")
    history = load_state(workflow.result("history"))
    if len(history["cycles"]) != workflow.settings()["cycles"]:
        raise ValueError("The retained baseline does not contain every requested cycle")
    physical = workflow.inventory(history["design"])
    reference = load_state(source / f"{cycle_point_keys(history)[0]}.json")
    chain = SparseChain(physical.chain, reference["case"]["depletion_nuclides"], physical.reactions)
    entry = restore_state(load_state(workflow.result("entry")))
    return physical, chain, FuelMap(physical), entry, history


def load_native_points(directory, physical, history):
    """Load fixed native responses for every history stage.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory containing or receiving the native state files.
    physical : IsotopicInventory
        Full-chain inventory binding and native case factory.
    history : mapping
        Retained cycle schedules, native observations, design values and endpoint fuel
        states.

    Returns
    -------
    dict of str to NativePoint
        State-keyed validated native responses for numerical replay.
    """
    from runtime.io import load_state

    points = {}
    for key in cycle_point_keys(history):
        path = Path(directory) / f"{key}.json"
        points[key] = NativePoint(load_state(path), physical.chain)
    return points


@interrupt_on_sigterm()
def prepare_baseline(workflow, *, reuse_baseline=False, native_only=False):
    """Prepare native evidence, calibration and residence gradients.

    Parameters
    ----------
    workflow : CycleCase
        Resolved case with model factories, input settings and result paths.
    reuse_baseline : bool, optional
        Restore and verify retained native evidence instead of generating a new history.
    native_only : bool, optional
        Stop after retaining the native baseline, before calibration and differentiation.

    Notes
    -----
    The current cycle engine requires a declared fixed boron concentration. Progress,
    failures, fit results and derivative checks are saved to case result paths. The
    derivative spans the finite fuel residence; it is not an equilibrium fixed-point
    derivative.
    """
    from coupling.calibration import calibrate_shared, calibration_identity, restore_shared
    from coupling.optimization import CycleObjective, check_residence_gradient_trace, keff_correction
    from runtime.io import save_json
    from optimization.design import set_parameters
    from fuel_cycle.residence import state_payload
    from runtime.autograd import GradientStorage

    workflow.prepare_outputs()
    config = workflow.settings()
    if config["boron_control"]:
        raise ValueError("This cycle engine currently supports a declared fixed boron concentration")
    torch.set_num_threads(1)
    started = time.perf_counter()
    status = {"status": "prepared", "pid": os.getpid(), "scope": config["scope"],
              "settings": workflow.settings(), "case_name": workflow.name}

    def progress(**changes):
        """Persist baseline progress and release unused native-stage memory.

        Parameters
        ----------
        **changes : object
            Named status, timing, stage or diagnostic updates.
        """
        status.update(changes)
        save_json(workflow.result("status"), status)
        if changes.get("status") == "native_completed":
            GradientStorage.release_unused()

    try:
        progress(design_evaluation="baseline")
        if reuse_baseline:
            physical, chain, mapping, entry, history = restore_baseline(workflow)
            progress(baseline_reused=True)
        else:
            physical, chain, mapping, entry, history = native_history(
                workflow, workflow.work / "baseline", config, progress,
                initialize=workflow.initialize_loading)
            save_state(workflow.result("history"), history)
            save_state(workflow.result("entry"), {"design": history["design"], **state_payload(entry)})
            retain_points(workflow, workflow.work / "baseline", history)
        if native_only:
            progress(status="native_history_completed", pid=None, native_pid=None,
                     total_wall_s=time.perf_counter() - started)
            return
        design = workflow.design()
        set_parameters(design, history["design"])
        fit_path = workflow.result("fit")
        fitted = json.loads(fit_path.read_text()) if fit_path.is_file() else {}
        identity = calibration_identity(workflow)
        if fitted.get("status") not in ("checked", "exploratory") or any(
                fitted.get(key) != value for key, value in identity.items()):
            calibrate_shared(workflow, physical, chain, mapping, entry, history, progress=progress)
        shared, fitted = restore_shared(workflow, physical, chain, mapping, entry, history, design)
        options = workflow.settings("optimization")
        objective = CycleObjective(shared, options)
        objective.keff_correction = keff_correction(
            history["observations"], fitted["selected"]["coupled"]["predictions"], options)
        trace = []
        progress(status="differentiating_full_residence", native_pid=None, stage=None)
        tick = time.perf_counter()
        output = objective(gradient_trace=trace)
        replay_s = time.perf_counter() - tick
        final = output["final_state"]
        native_final = history["cycles"][-1]["end_state"]
        inventory = np.asarray(native_final["inventory"])
        energy = np.asarray(native_final["energy_j"])
        n_error = float(np.max(abs(final.inventory.detach().numpy() - inventory)
                               / np.maximum(abs(inventory), DENSITY_FLOOR)))
        energy_error = float(np.max(abs(final.energy_j.detach().numpy() - energy)
                                    / np.maximum(abs(energy), 1.)))
        tick = time.perf_counter()
        output["loss"].backward()
        backward_s = time.perf_counter() - tick
        gradient = {name: p.grad.detach().clone() for name, p in design.named_parameters()}
        if not all(torch.isfinite(value).all() for value in gradient.values()):
            raise ArithmeticError("The coupled cycle gradient is nonfinite")
        trace_check = check_residence_gradient_trace(trace, history, options)
        checks = {"status": "gradients_computed", "gradient_graph_passed": True,
                  "settings": workflow.settings(), "case_name": workflow.name,
                  "calibration_sha256": sha(fit_path),
                  "optimization_settings": workflow.settings("optimization"),
                  "objective": options["objective"],
                  "case_definition": identity["case_definition"],
                  "native_replay_inventory_relative_error": n_error,
                  "native_replay_energy_relative_error": energy_error,
                  "gradient": {name: value.tolist() for name, value in gradient.items()},
                  "inventory_gradient_flow": trace, "gradient_trace_check": trace_check,
                  "replay_wall_s": replay_s, "backward_wall_s": backward_s,
                  "baseline_metrics": objective.latest["metrics"],
                  "cycle_days": objective.latest["cycle_days"],
                  "shared_calibration_parameters": len(fitted["selected"]["parameter_names"]),
                  "field_quality_passed": fitted["field_quality_passed"],
                  "method": "Differentiate Bateman depletion, diffusion, carried energy and synchronized loading through every declared cycle from fresh fuel. Closure factors and native within-group spectra remain fixed locally. This is a finite-residence derivative, not an equilibrium fixed-point derivative."}
        save_json(workflow.result("gradient_checks"), checks)
        progress(status="full_residence_gradients_computed", pid=None, native_pid=None,
                 calibration_status=fitted["status"], field_quality_passed=fitted["field_quality_passed"],
                 total_wall_s=time.perf_counter() - started)
    except BaseException as error:
        progress(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                 pid=None, native_pid=None, error=f"{type(error).__name__}: {error}")
        raise


def check_replay(workflow):
    """Check depletion and transfers using the exact retained native rates.

    Parameters
    ----------
    workflow : CycleCase
        Resolved case with model factories, input settings and result paths.

    Raises
    ------
    ArithmeticError
        The maximum inventory error, scaled by the declared density floor, reaches the 1e-10
        replay limit.

    Notes
    -----
    A replay-check record is written before failure is raised. This check isolates numerical
    depletion and fuel transfer; it does not validate surrogate accuracy or design
    gradients.
    """
    from runtime.processes import native_library_path
    import numpy as np
    import torch

    os.environ["OPENMC_REFERENCE_LIBRARY"] = str(native_library_path())
    from runtime.io import sha
    from diffusion_surrogate.solver import tensor
    from fuel_cycle.residence import replay_history

    torch.set_num_threads(1)
    physical, chain, mapping, entry, history = restore_baseline(workflow)
    design = workflow.design()
    with torch.no_grad():
        for name, value in history["design"].items():
            getattr(design, name).copy_(tensor(value))
    points = load_native_points(workflow.native_states(), physical, history)
    observations = []
    for key, point in points.items():
        original = point.evaluate

        def observe(inventory, boron, key=key, point=point, original=original):
            """Record stage inventory differences while returning native rates.

            Parameters
            ----------
            inventory : torch.Tensor
                Region-by-chain CPU float64 atom densities in atoms/cm³.
            boron : float or torch.Tensor
                Boron-10 concentration in mg per kg of unborated moderator.
            key : str, optional
                Stable native state key, including its cycle and stage where applicable.
            point : NativePoint, optional
                Captured native reference for this stage.
            original : callable, optional
                Captured unwrapped native evaluation method.

            Returns
            -------
            dict
                Original fixed native stage output.
            """
            output = original(inventory, boron)
            difference = (inventory - point.reference_inventory).detach().numpy()
            reference = point.reference_inventory.numpy()
            scaled = abs(difference) / np.maximum(abs(reference), DENSITY_FLOOR)
            zone, isotope = np.unravel_index(np.argmax(scaled), scaled.shape)
            native_rates = point.native_rates_by_group.sum(-1).numpy()
            rates = output["rates"].detach().numpy()
            row = {
                "stage": key,
                "inventory_relative_l1": float(abs(difference).sum() / abs(reference).sum()),
                "inventory_max_scaled_error": float(scaled[zone, isotope]),
                "worst_zone": physical.labels[zone],
                "worst_nuclide": point.nuclides[isotope],
                "worst_reference_atoms_cm3": float(reference[zone, isotope]),
                "worst_absolute_difference_atoms_cm3": float(difference[zone, isotope]),
                "rates_max_scaled_error": float(
                    np.max(abs(rates - native_rates) / np.maximum(abs(native_rates), 1e-30))
                ),
            }
            observations.append(row)
            return output

        point.evaluate = observe
    with torch.no_grad():
        metrics, _, _, _ = replay_history(
            mapping, chain, design, history, points, workflow.settings()
        )
    maximum_error = max((row["inventory_max_scaled_error"] for row in observations))
    result = {
        "status": "checked" if maximum_error < 1e-10 else "failed",
        "settings": workflow.settings(),
        "case_name": workflow.name,
        "history_sha256": sha(workflow.result("history")),
        "design": history["design"],
        "completed_cycles": len(history["cycles"]),
        "density_scale_floor_atoms_cm3": DENSITY_FLOOR,
        "stages": observations,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "maximum_inventory_scaled_error": maximum_error,
        "scope": "Numerical replay with the exact native rates, without any diffusion fits. This verifies depletion and fuel transfers, not surrogate accuracy or design gradients.",
    }
    workflow.result("replay_checks").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "checked":
        raise ArithmeticError("Native-rate history replay differs from the retained trajectory")


def check_sampling(workflow):
    """Check retained single-seed native tally precision.

    Parameters
    ----------
    workflow : CycleCase
        Resolved case with model factories, input settings and result paths.

    Raises
    ------
    ValueError
        A state uses different sampling or cannot form the required complete generation
        blocks.
    ArithmeticError
        A configured keff or fuel-power precision limit is exceeded.

    Notes
    -----
    The saved record also includes entropy and active-half drift diagnostics. No additional
    native run is performed; these checks do not establish mesh, timestep or equilibrium
    convergence.
    """
    from runtime.processes import native_library_path
    import numpy as np

    os.environ["OPENMC_REFERENCE_LIBRARY"] = str(native_library_path())
    from runtime.io import sha
    from runtime.io import load_state

    config = workflow.settings()
    limits = config["sampling_checks"]
    history = load_state(workflow.result("history"))
    rows = []
    for key in cycle_point_keys(history):
        state = load_state(workflow.native_states() / f"{key}.json")
        if state["case"]["sampling"] != config["sampling"]:
            raise ValueError("A native state does not use the declared single-seed sampling")
        result = state["result"]
        fuel = result["depletion_region_indices"]
        power = result["observables"]["region_power_W"]
        relative = np.asarray(power["scale"])[fuel] / np.asarray(power["mean"])[fuel]
        entropy = np.asarray(result["entropy"])
        generation = np.asarray(result["k_generation"])[config["sampling"]["inactive"] :]
        block_size = config.get("sampling_block_size", 4)
        if len(generation) % block_size or len(generation) < 4 * block_size:
            raise ValueError("Active generations must form at least four complete sampling blocks")
        blocks = generation.reshape(-1, block_size).mean(1)
        a, b = np.array_split(blocks, 2)
        scale = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        row = {
            "stage": key,
            "keff_standard_error": result["combined_keff"][1],
            "power_relative_se_rms": float(np.sqrt(np.mean(relative**2))),
            "power_relative_se_max": float(relative.max()),
            "active_keff_half_difference_scaled": float(
                abs(a.mean() - b.mean()) / max(scale, 1e-30)
            ),
            "entropy_inactive_tail_mean": float(
                entropy[config["sampling"]["inactive"] - 20 : config["sampling"]["inactive"]].mean()
            ),
            "entropy_active_tail_mean": float(entropy[-20:].mean()),
            "physical_inventory_checks": result["physical_inventory_checks"],
        }
        row["precision_passed"] = (
            row["keff_standard_error"] <= limits["maximum_keff_standard_error"]
            and row["power_relative_se_rms"] <= limits["maximum_power_relative_se_rms"]
            and (row["power_relative_se_max"] <= limits["maximum_power_relative_se"])
        )
        rows.append(row)
    record = {
        "status": "checked" if all((r["precision_passed"] for r in rows)) else "failed",
        "settings": workflow.settings(),
        "case_name": workflow.name,
        "history_sha256": sha(workflow.result("history")),
        "completed_cycles": len(history["cycles"]),
        "requested_cycles": config["cycles"],
        "sampling": config["sampling"],
        "limits": limits,
        "sampling_allowance": config.get("sampling_allowance"),
        "states": rows,
        "seed": config["sampling"]["seed"],
        "maximum_keff_standard_error": max((r["keff_standard_error"] for r in rows)),
        "maximum_power_relative_se_rms": max((r["power_relative_se_rms"] for r in rows)),
        "maximum_power_relative_se": max((r["power_relative_se_max"] for r in rows)),
        "scope": "Single-seed statewise tally precision with entropy and active-half drift diagnostics. No second-seed run. This is not mesh, timestep or equilibrium qualification.",
    }
    workflow.result("sampling_check").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "status",
                    "seed",
                    "maximum_keff_standard_error",
                    "maximum_power_relative_se_rms",
                    "maximum_power_relative_se",
                )
            },
            indent=2,
        )
    )
    if record["status"] != "checked":
        raise ArithmeticError("Native tally precision check failed")
