"""Cache OpenMC's linear histogram-to-rate map at fixed data and temperature."""
import ctypes as ct
import json
from pathlib import Path
import time

import numpy as np
import openmc.data
from diffusion_surrogate.native3d import collapsed_depletion


def collapse_cached(dll, case, spectrum=None):
    """Reuse native histogram-to-rate kernels for matching processing identities.

    Parameters
    ----------
    dll : ctypes.CDLL
        Initialized native OpenMC shared library with the current model and tally data
        loaded.
    case : dict
        Native model description with ordered regions, sampling, energy groups and
        depletion-tally settings.
    spectrum : dict or None, optional
        Saved energy_eV edges and fuel-region-by-bin flux scores; None reads the current
        native flux tally.

    Returns
    -------
    dict
        Collapsed and coarse-group reaction scores, flux histogram, energy edges, kernel
        identity and direct-collapse agreement check.

    Raises
    ------
    RuntimeError
        OpenMC cannot load a required parent nuclide or collapse one energy bin.
    ArithmeticError
        Cached kernels fail direct-collapse agreement or group-rate conservation.

    Notes
    -----
    Cache identity includes temperatures, energy grid, nuclide/reaction ordering and
    evaluated-data/native-library paths, sizes and modification times. New kernels are
    checked against direct native collapse before use.
    """
    path = Path(case["depletion_kernel_cache"])
    regions = [r for r in case["regions"] if r["depletable"]]
    temperatures = sorted({r["temperature_K"] for r in regions})
    empty = dict(case, depletion_nuclides=[], depletion_flux_refinement_check=False)
    observed = collapsed_depletion(dll, empty, spectrum=spectrum)
    edges, flux = observed["energy_eV"], observed["flux"]
    library = openmc.data.DataLibrary.from_xml(case["cross_sections"])
    paths = {name: Path(item["path"]) for item in library if item["type"] == "neutron" for name in item["materials"]}
    files = {name: (str(paths[name]), paths[name].stat().st_size, paths[name].stat().st_mtime_ns)
             for name in case["depletion_nuclides"]}
    from runtime.processes import native_library_path
    binary = native_library_path()
    identity = {"version": 1, "temperatures_K": temperatures, "energy_eV": edges.tolist(),
                "nuclides": case["depletion_nuclides"], "mts": case["depletion_reaction_mts"],
                "data_files": files, "library": [str(binary), binary.stat().st_size, binary.stat().st_mtime_ns]}
    configuration = json.dumps(identity, sort_keys=True)
    kernel, checks = None, None
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["identity"]) == configuration:
                kernel = saved["kernel"]
                checks = json.loads(str(saved["checks"]))
    started = time.perf_counter()
    if kernel is None:
        pointer = ct.POINTER(ct.c_double)
        dll.openmc_load_nuclide.argtypes = [ct.c_char_p, pointer, ct.c_int]
        dll.openmc_get_nuclide_index.argtypes = [ct.c_char_p, ct.POINTER(ct.c_int)]
        dll.openmc_nuclide_collapse_rate.argtypes = [ct.c_int, ct.c_int, ct.c_double, pointer, pointer, ct.c_int, pointer]
        values = np.ascontiguousarray(temperatures, dtype=float)
        one = np.ones(1)
        kernel = np.zeros((len(temperatures), len(case["depletion_nuclides"]),
                           len(case["depletion_reactions"]), len(edges) - 1))
        for parent, name in enumerate(case["depletion_nuclides"]):
            code = dll.openmc_load_nuclide(name.encode(), values.ctypes.data_as(pointer), len(values))
            if code:
                raise RuntimeError(f"Cannot load native rate parent {name}: {code}")
            index = ct.c_int()
            if dll.openmc_get_nuclide_index(name.encode(), ct.byref(index)):
                raise RuntimeError(f"Missing loaded native rate parent {name}")
            for ti, temperature in enumerate(temperatures):
                for reaction, mt in enumerate(case["depletion_reaction_mts"]):
                    for group in range(len(edges) - 1):
                        response = ct.c_double()
                        code = dll.openmc_nuclide_collapse_rate(index, mt, temperature,
                            edges[group:group + 2].ctypes.data_as(pointer), one.ctypes.data_as(pointer), 1,
                            ct.byref(response))
                        if code:
                            raise RuntimeError(f"Native one-bin collapse failed: {name}, MT{mt}, {group}: {code}")
                        kernel[ti, parent, reaction, group] = response.value
        direct = collapsed_depletion(dll, dict(case, depletion_flux_refinement_check=False), spectrum=observed)
        zones = kernel[[temperatures.index(r["temperature_K"]) for r in regions]]
        candidate = np.einsum("zb,zprb->zpr", flux, zones, optimize=True)
        scale = np.maximum(abs(direct["rates"]), 1e-30)
        error = float(np.max(abs(candidate - direct["rates"]) / scale))
        if not np.allclose(candidate, direct["rates"], rtol=5e-11, atol=1e-30):
            raise ArithmeticError(f"Cached native rate kernel differs from direct collapse: {error:g}")
        checks = {"native_collapse_max_relative_error": error, "build_and_check_wall_s": time.perf_counter() - started,
                  "scope": "Native unit-bin responses reproduce original histogram collapse; no fitted cross sections."}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, kernel=kernel, identity=configuration, checks=json.dumps(checks))
        temporary.replace(path)
    zones = kernel[[temperatures.index(r["temperature_K"]) for r in regions]]
    rates = np.einsum("zb,zprb->zpr", flux, zones, optimize=True)
    by_group = []
    for low, high in zip(case["group_edges_eV"][:-1], case["group_edges_eV"][1:]):
        fraction = np.maximum(0., np.minimum(edges[1:], high) - np.maximum(edges[:-1], low)) / np.diff(edges)
        by_group.append(np.einsum("zb,zprb->zpr", flux * fraction, zones, optimize=True))
    grouped = np.stack(by_group, axis=-1)
    if not np.allclose(grouped.sum(-1), rates, rtol=1e-12, atol=1e-30):
        raise ArithmeticError("Cached two-group decomposition lost reaction rate")
    return {"rates": rates, "rates_by_group": grouped, "flux": flux, "energy_eV": edges,
            "kernel_identity": configuration, "kernel_check_relative": checks["native_collapse_max_relative_error"]}
