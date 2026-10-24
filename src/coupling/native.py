"""Isolated native OpenMC evaluations for fuel-cycle histories."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from runtime.io import load_state, save_state


def evaluate_native(physical, inventory, boron, sampling, path, directory, *, progress=None):
    """Reuse an identical native point or evaluate it in a fresh process.

    Parameters
    ----------
    physical : IsotopicInventory
        Full-chain inventory binding, native case factory and live fresh-fuel recipe.
    inventory : array_like
        Full-chain fuel atom densities in atoms/cm³.
    boron : float or torch.Tensor
        Boron-10 concentration in mg per kg of unborated moderator.
    sampling : mapping
        Native particle, batch, seed and thread settings.
    path : str or pathlib.Path
        Retained state JSON path with its paired array file.
    directory : str or pathlib.Path
        Working directory for native inputs, logs and transient results.
    progress : callable or None, optional
        Optional callback receiving process status, criticality and timing updates.

    Returns
    -------
    dict
        Retained native case and result with the exact full-chain inventory and identities.

    Raises
    ------
    ValueError
        A matching cutoff transport case hides a different full-chain inventory.
    RuntimeError
        The isolated native process exits unsuccessfully.

    Notes
    -----
    OpenMC owns process-global library state. A separate process permits safe
    temperature/material changes without sharing mutable OpenMC state between evaluations.
    """
    from runtime.io import serializable
    path, directory = Path(path), Path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    case = physical.case(inventory, boron, sampling=sampling, derivatives=True)
    if path.is_file():
        state = load_state(path)
        if serializable(state["case"]) == serializable(case):
            if not np.array_equal(state["fuel_inventory_atoms_cm3"], inventory):
                raise ValueError("Matching transport cutoff hides a different full-chain inventory")
            if progress:
                progress(status="native_reused", native_pid=None, keff=state["result"]["combined_keff"])
            return state
    case_path = directory / "case.json"
    case_path.write_text(json.dumps(serializable(case), indent=2, allow_nan=False) + "\n")
    environment = dict(os.environ, CORE_NATIVE_DIRECTORY=str(directory.resolve()),
                       CORE_NATIVE_DRIVER=physical.native_driver,
                       OMP_NUM_THREADS=str(sampling["threads"]),
                       OPENBLAS_NUM_THREADS="1")
    process = None
    started = time.perf_counter()
    try:
        with (directory / "native.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "coupling.native"], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            if progress:
                progress(status="native_running", native_pid=process.pid, boron_mg_B10_per_kg=boron,
                         transported_fuel_nuclides=len(case["material_response"]["nuclides"]))
            print(f"OpenMC PID {process.pid}: {path.stem}, B={boron:.6g} mg B10/kg", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"OpenMC failed ({code}); inspect {directory / 'native.log'}")
        result = load_state(directory / "result.json")
        state = {"case": case, "result": result,
                 "fuel_inventory_atoms_cm3": np.asarray(inventory), "nuclides": physical.names,
                 "fuel_labels": physical.labels}
        save_state(path, state)
        if progress:
            progress(status="native_completed", native_pid=None, keff=result["combined_keff"],
                     native_wall_s=time.perf_counter() - started)
        print(f"  k={result['combined_keff'][0]:.6f} +/- {result['combined_keff'][1]:.6f}; {result['wall_s']:.1f} s", flush=True)
        return state
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()


def main():
    """Run the environment-selected native driver and save its result.

    Notes
    -----
    CORE_NATIVE_DRIVER selects module:function and CORE_NATIVE_DIRECTORY supplies the
    working folder containing case.json. This is the subprocess entry point used by
    evaluate_native.
    """
    from runtime.processes import native_library_path
    os.environ["OPENMC_REFERENCE_LIBRARY"] = str(native_library_path())
    from importlib import import_module
    module, function = os.environ["CORE_NATIVE_DRIVER"].split(":")
    run = getattr(import_module(module), function)
    directory = Path(os.environ["CORE_NATIVE_DIRECTORY"])
    case = json.loads((directory / "case.json").read_text())
    save_state(directory / "result.json", run(case, directory))


if __name__ == "__main__":
    main()
