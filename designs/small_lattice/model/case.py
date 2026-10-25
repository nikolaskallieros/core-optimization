"""Run the small lattice through native histories and calibrated shape optimization."""

from datetime import datetime, timezone
from pathlib import Path
import time
import tomllib

import numpy as np
from scipy.linalg import expm
import torch

from coupling.depletion import DepletionHistory, DepletionResponse
from coupling.multigroup import MultigroupTransport
from diffusion_surrogate import CalibratedDiffusion, SparseDiffusion3D
from geometry.diffusion import discretize
from optimization.design import Snapshot, optimize
from optimization.fitting import Target, load_fit
from runtime.io import save_json
from runtime.paths import RUNS

from ..build_geometry import build_geometry, pin_centers
from .design import Lattice
from .materials import Materials, build_chain

DESIGN_ROOT = Path(__file__).resolve().parents[1]
INPUTS = DESIGN_ROOT / "inputs/model.toml"


def make_history(settings, mesh, materials, chain, native=None):
    """Wire shared material, transport and depletion interfaces for this design.

    Parameters
    ----------
    settings : dict
        Resolved model.toml configuration.
    mesh : diffusion_surrogate.geometry.DiffusionGeometry
        Live material mapping on the fixed background mesh.
    materials : Materials
        Manufactured microscopic data and fresh isotope densities.
    chain : depletion.sparse.SparseChain
        Compiled four-isotope demonstration chain.
    native : coupling.multigroup.MultigroupTransport or None, optional
        Native evaluator; None builds a calibrated diffusion surrogate.

    Returns
    -------
    coupling.depletion.DepletionHistory
        Fresh-start prescribed-power history with a shared solver interface.
    """
    reference = None
    if native is None:
        reference = materials.macroscopic(materials.initial)
        basis = torch.ones((1, len(mesh.region_labels), 1), dtype=torch.float64)
        native = CalibratedDiffusion(SparseDiffusion3D(mesh.grid), mesh.fractions,
            reference, {"diffusion": basis}, ("diffusion",), power_W=settings["history"]["power_W"])
    response = DepletionResponse(native, materials.macroscopic, materials.rates,
                                 range(materials.regions), reference=reference)
    return DepletionHistory(response, chain,
        lambda: materials.fresh_state((mesh.fractions().T @ mesh.grid.volumes)[:materials.regions]),
        settings["history"]["durations_days"])


def check_numerics(design, model, refined, settings):
    """Check the full-history derivative, depletion kernel and numerical resolution.

    Parameters
    ----------
    design : Lattice
        Selected radius parameter.
    model, refined : coupling.depletion.DepletionHistory
        Calibrated histories with identical coefficients and different meshes.
    settings : dict
        Input criteria, power and history intervals.

    Returns
    -------
    dict
        Measured errors and final surrogate responses, after all criteria pass.
    """
    criteria = settings["verification"]
    output = model()
    derivative, = torch.autograd.grad(output["keff"], design.coordinate)
    original = design.coordinate.detach().clone()
    step = criteria["gradient_coordinate_step"]
    with torch.no_grad():
        values = []
        for increment in (-step, step):
            design.coordinate.copy_(original + increment)
            values.append(float(model()["keff"]))
        design.coordinate.copy_(original)
        fine = refined()
    finite_difference = (values[1] - values[0]) / (2 * step)
    gradient_error = abs(float(derivative) / finite_difference - 1)
    refinement_error = abs(float(fine["keff"]) - float(output["keff"].detach()))
    expected_energy = settings["history"]["power_W"] * sum(settings["history"]["durations_days"]) * 86400
    energy_error = abs(float(output["energy_j"].detach().sum()) / expected_energy - 1)

    # Independent dense exponential for a frozen-rate predictor step; the actual
    # coupled history above uses the shared CE/CM and sparse CRAM48 implementation.
    with torch.no_grad():
        state = model.initial_state()
        rates = model.response(state.inventory)["rates"]
        seconds = settings["history"]["durations_days"][0] * 86400 / 2
        actual = model.chain(state.inventory, rates, seconds).numpy()
        entries = model.chain.matrix_values(rates).numpy()
        expected = []
        for coefficients, atoms in zip(entries, state.inventory.numpy()):
            matrix = np.zeros((model.chain.size, model.chain.size))
            matrix[model.chain.rows.numpy(), model.chain.columns.numpy()] = coefficients
            expected.append(expm(matrix * seconds) @ atoms)
        expected = np.asarray(expected)
        depletion_error = float(np.max(np.abs(actual - expected)
                                / np.maximum(np.abs(expected), state.inventory.max().item() * 1e-12)))
    if gradient_error > criteria["gradient_rtol"]:
        raise AssertionError(f"Full-history radius derivative differs from centered differences: {gradient_error}")
    if refinement_error > criteria["refined_keff_tolerance"]:
        raise AssertionError(f"Final k-effective changes too much on the refined mesh: {refinement_error}")
    if depletion_error > criteria["depletion_rtol"] or energy_error > criteria["energy_rtol"]:
        raise AssertionError(f"Depletion/energy check failed: {depletion_error}, {energy_error}")
    radius_slope = (design.upper - design.lower) * original.sigmoid() * (1 - original.sigmoid())
    return {"gradient_relative_error": gradient_error,
            "dkeff_dradius_per_cm": float(derivative / radius_slope),
            "refined_keff_change": refinement_error, "refined_keff": float(fine["keff"]),
            "depletion_exponential_relative_error": depletion_error, "energy_relative_error": energy_error,
            "surrogate_history": {key: value.detach().tolist() for key, value in output.items()}}


def run(inputs=INPUTS):
    """Execute two native-verified design proposals in a fresh local run directory.

    Parameters
    ----------
    inputs : str or pathlib.Path, optional
        One TOML file defining the lattice, materials, history and objective.

    Returns
    -------
    dict
        Selected native result, fitted coefficients, design history, timings and
        numerical verification. Raw native states remain in the reported run folder.
    """
    started = time.perf_counter()
    inputs = Path(inputs)
    settings = tomllib.loads(inputs.read_text())
    options, criteria = settings["optimization"], settings["verification"]
    directory = RUNS / DESIGN_ROOT.name / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    (directory / "model.toml").write_text(inputs.read_text())
    (directory / "goals.md").write_text((DESIGN_ROOT / "goals.md").read_text())
    print(f"Run: {directory}", flush=True)
    design = Lattice(settings["geometry"])
    materials = Materials(settings["materials"], int(np.prod(settings["geometry"]["shape"])))
    chain = build_chain(settings["chain"], settings["materials"])
    cad = lambda: build_geometry(settings["geometry"], design.radius_cm)
    mesh = discretize(cad, settings["mesh"]["subdivisions"])
    references = []

    def detailed(current, previous):
        """Run a new native history at the current physical radius.

        Parameters
        ----------
        current : Lattice
            Candidate design supplied by the shared optimization loop.
        previous : optimization.design.Snapshot or None
            Previous accepted reference; every history here starts fresh.

        Returns
        -------
        optimization.design.Snapshot
            Current native observations, fitting target and retained metadata.
        """
        native = MultigroupTransport(cad(), directory / f"native_{len(references):02d}",
            settings["materials"]["energy_edges_eV"], settings["openmc"],
            power_W=settings["history"]["power_W"])
        mapped_volumes = mesh.fractions().T @ mesh.grid.volumes
        volume_error = float(((mapped_volumes - native.volumes) / native.volumes).abs().max())
        if volume_error > criteria["volume_rtol"]:
            raise AssertionError(f"Diffusion/CAD material volumes disagree: {volume_error}")
        for index, xy in enumerate(pin_centers(settings["geometry"])):
            cell = native.adapted.geometry.find((*xy.tolist(), settings["geometry"]["height_cm"] / 2))[-1]
            if cell.id != native.adapted.cells[index].id:
                raise AssertionError(f"Native material identity changed at fuel pin {index}")
        history = make_history(settings, mesh, materials, chain, native)()
        energy = settings["history"]["power_W"] * sum(settings["history"]["durations_days"]) * 86400
        energy_error = abs(float(history["energy_j"].sum()) / energy - 1)
        if energy_error > criteria["energy_rtol"]:
            raise AssertionError(f"Native history energy accounting failed: {energy_error}")
        record = {"radius_cm": float(current.radius_cm), "directory": str(native.directory),
                  "volume_relative_error": volume_error, "energy_relative_error": energy_error,
                  "native_states": native.records,
                  "history": {key: value.detach().tolist() for key, value in history.items()}}
        save_json(native.directory / "history.json", record)
        references.append(record)
        print(f"Native {len(references) - 1}: radius {record['radius_cm']:.5f} cm; "
              f"final k {float(history['keff']):.6f} +/- {float(history['standard_error']):.6f}", flush=True)
        reference = {"observables": {"keff": {"mean": float(history["keff"]),
                                              "scale": options["calibration_scale"]}}}
        return Snapshot(history, reference, metadata=record)

    def accept(current, candidate, proposal):
        """Retain a native improvement or a candidate satisfying the target.

        Parameters
        ----------
        current, candidate : optimization.design.Snapshot
            Independently computed native histories.
        proposal : dict
            Shared optimizer's candidate record.

        Returns
        -------
        dict
            Acceptance flag and the candidate's absolute native target error.
        """
        old_error = abs(float(current.outputs["keff"]) - options["target_keff"])
        error = abs(float(candidate.outputs["keff"]) - options["target_keff"])
        tolerance = max(criteria["native_absolute_tolerance"],
                        criteria["native_sigma_multiplier"] * float(candidate.outputs["standard_error"]))
        return {"accepted": error < old_error or error <= tolerance, "native_target_error": error}

    optimizer = torch.optim.LBFGS(design.parameters(), max_iter=options["design_iterations"],
        tolerance_grad=1e-10, tolerance_change=1e-14, line_search_fn="strong_wolfe")
    result = optimize(design, detailed, lambda current, reference: make_history(settings, mesh, materials, chain),
        [Target("keff", transform="identity")], lambda output: (output["keff"] - options["target_keff"])**2,
        optimizer=optimizer, steps=options["design_updates"], accept=accept, keep_best=True,
        fit_options={"steps": options["fit_iterations"], "jacobian_mode": "reverse",
                     "optimizer_options": {"tolerance_grad": 1e-10, "tolerance_change": 1e-14}},
        fit_accept=lambda report: abs(report["predictions"]["keff"]["prediction"][0]
            - report["predictions"]["keff"]["reference"][0]) <= criteria["calibration_tolerance"])
    final = result["snapshot"]
    final_error = abs(float(final.outputs["keff"]) - options["target_keff"])
    standard_error = float(final.outputs["standard_error"])
    tolerance = max(criteria["native_absolute_tolerance"], criteria["native_sigma_multiplier"] * standard_error)
    if final_error > tolerance or standard_error > criteria["native_maximum_standard_error"]:
        raise AssertionError(f"Final native target/precision failed: error {final_error}, standard error {standard_error}")
    refined_mesh = discretize(cad, settings["mesh"]["refined_subdivisions"])
    refined = make_history(settings, refined_mesh, materials, chain)
    load_fit(refined, result["selected_fit"])
    checks = check_numerics(design, result["model"], refined, settings)
    record = {"passed": True, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Manufactured one-group CAD/OpenMC/diffusion/depletion software demonstration",
              "directory": str(directory), "wall_s": time.perf_counter() - started,
              "initial_radius_cm": references[0]["radius_cm"], "final_radius_cm": float(design.radius_cm.detach()),
              "initial_native_keff": references[0]["history"]["keff"], "final_native_keff": float(final.outputs["keff"]),
              "final_native_standard_error": standard_error, "native_target_tolerance": tolerance,
              "references": references, "selected_native_directory": final.metadata["directory"],
              "fits": result["fits"], "proposals": result["proposals"], "selected_fit": result["selected_fit"],
              "checks": checks}
    save_json(directory / "verification.json", record)
    print(f"Passed: radius {record['final_radius_cm']:.5f} cm; final OpenMC k "
          f"{record['final_native_keff']:.6f} +/- {standard_error:.6f}; {record['wall_s']:.1f} s", flush=True)
    return record
