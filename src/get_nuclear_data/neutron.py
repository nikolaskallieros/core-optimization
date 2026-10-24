"""Process declared ENDF/B-VII.1 neutron and thermal tables with NJOY."""
import contextlib
import json
import os
from pathlib import Path
import sys
import time
from get_nuclear_data import endf
from get_nuclear_data.verification import dataset_hashes
from get_nuclear_data.processing import processing_context
from runtime.io import sha, save_json
from runtime.processes import peak_rss_mib


def prepare_neutron_data(directory, specification, source_manifest, *, thermal_scattering,
                         njoy=None, temperature_tolerance=.05):
    """Process declared neutron and thermal-scattering evaluations with NJOY.

    Parameters
    ----------
    directory : str or pathlib.Path
        Destination nuclear-data cache for tables, NJOY decks, logs and processing receipts.
    specification : dict
        Explicit neutron/thermal temperatures in K and relative NJOY error tolerance.
    source_manifest : str or pathlib.Path
        JSON source manifest containing expected evaluation checksums.
    thermal_scattering : dict
        Scattering-table names mapped to their neutron nuclide and thermal archive member.
    njoy : str or pathlib.Path or None, optional
        NJOY executable; None selects bin/njoy in the active environment.
    temperature_tolerance : float, optional
        Allowed processed-temperature discrepancy in K.

    Raises
    ------
    ValueError
        A neutron temperature selection cannot support the declared interpolation.
    RuntimeError
        Existing processing identities, source content, processed names or temperatures
        disagree.

    Notes
    -----
    Evaluated-data checksums and processing settings identify reusable data. Generated
    decks, scratch files and receipts belong to the destination cache.
    """
    directory, source_manifest = Path(directory), Path(source_manifest)
    njoy = Path(njoy) if njoy is not None else Path(sys.prefix) / "bin/njoy"
    receipt = directory / "manifest.json"
    with processing_context(directory, receipt):
        return _prepare(directory, specification, source_manifest, thermal_scattering,
                        njoy, temperature_tolerance, receipt)


def _prepare(directory, specification, source_manifest, thermal_scattering,
             njoy, temperature_tolerance, receipt):
    """Prepare selected evaluations within the managed processing context.

    Parameters
    ----------
    directory : str or pathlib.Path
        Destination nuclear-data cache for tables, NJOY decks, logs and processing receipts.
    specification : dict
        Explicit neutron/thermal temperatures in K and relative NJOY error tolerance.
    source_manifest : str or pathlib.Path
        JSON source manifest containing expected evaluation checksums.
    thermal_scattering : dict
        Scattering-table names mapped to their neutron nuclide and thermal archive member.
    njoy : str or pathlib.Path or None
        NJOY executable; None selects bin/njoy in the active environment.
    temperature_tolerance : float
        Allowed processed-temperature discrepancy in K.
    receipt : pathlib.Path
        Processing-manifest destination updated after completed stages.

    Notes
    -----
    Writes processed HDF5 data, cross_sections.xml and a resumable manifest. Temperature
    extensions must preserve every previously retained HDF5 dataset.
    """
    decks = directory / "decks"
    os.environ["OMP_NUM_THREADS"] = "1"
    import openmc.data

    processing = dict(specification, thermal_scattering=thermal_scattering)
    directory.mkdir(parents=True, exist_ok=True)
    decks.mkdir(exist_ok=True)
    scratch = directory / "scratch"
    scratch.mkdir(exist_ok=True)
    sources = directory / "endf"
    temperatures = specification["neutron_temperatures_K"]
    members = {}
    for name in temperatures:
        symbol = "".join(c for c in name if c.isalpha())
        mass = int("".join(c for c in name if c.isdigit()))
        members[name] = f"neutrons/n-{openmc.data.ATOMIC_NUMBER[symbol]:03d}_{symbol}_{mass:03d}.endf"
    archive_members = {"neutrons": members, "thermal_scatt": {
        name: choice["member"] for name, choice in thermal_scattering.items()}}
    package = next((njoy.parent.parent / "conda-meta").glob("njoy2016-*.json"), None)
    njoy_record = json.loads(package.read_text()) if package is not None else None
    record = json.loads(receipt.read_text()) if receipt.exists() else {
        "status": "preparing", "library": "ENDF/B-VII.1",
        "sources": json.loads(source_manifest.read_text())["sources"], "processed": {},
        "processing_specification": processing,
        "njoy_package": (f"njoy2016={njoy_record['version']}={njoy_record['build']}"
                         if njoy_record is not None else None),
        "openmc_version": openmc.__version__}
    current = {k: v for k, v in processing.items() if k != "neutron_temperatures_K"}
    previous = {k: record["processing_specification"][k] for k in current}
    if previous != current:
        raise RuntimeError("Existing nuclear data has a different processing specification")
    if any(len(values) < 2 for values in temperatures.values()):
        raise ValueError("OpenMC interpolation requires two neutron temperatures per nuclide; a single table switches the global method to nearest, including S(a,b).")
    record["processing_specification"] = processing
    record.update(status="downloading", pid=os.getpid())
    save_json(receipt, record)
    endf.fetch_sources(record, members=archive_members, sources=sources, receipt=receipt)
    for name in (*members, *archive_members["thermal_scatt"]):
        source = sources / f"{name}.endf"
        evaluation = openmc.data.endf.Evaluation(source)
        if tuple(evaluation.info["library"]) != ("ENDF/B", 7, 1):
            # The NNDC VII.1 Zr evaluations carry NLIB=10, which OpenMC labels
            # Unknown. Their release archive, MF1 header and target identify
            # them as VII.1; retain this metadata discrepancy explicitly.
            header = source.read_text().splitlines()[:20]
            zirconium_header = (name in {"Zr90", "Zr91", "Zr92", "Zr94", "Zr96"}
                                and tuple(evaluation.info["library"]) == ("Unknown", 7, 1)
                                and evaluation.target["atomic_number"] == 40
                                and evaluation.target["mass_number"] == int(name[2:])
                                and any("----ENDF/B-VII.1" in line for line in header))
            if not zirconium_header:
                raise RuntimeError(f"Unexpected evaluated library for {name}")
            record["sources"][name]["library_metadata_note"] = "NLIB=10 is Unknown to OpenMC; NNDC VII.1 archive and MF1 header identify ENDF/B-VII.1."
        target = directory / f"{name}.h5"
        desired = temperatures.get(name, specification["thermal_temperatures_K"])
        if name in record["processed"]:
            if sha(target) != record["processed"][name]["sha256"]:
                raise RuntimeError(f"Existing processed data failed verification: {name}")
            actual = record["processed"][name]["temperature_K"]
            if len(actual) == len(desired) and all(abs(a-b) < temperature_tolerance
                                                  for a, b in zip(sorted(actual), sorted(desired))):
                continue
        elif target.exists():
            raise RuntimeError(f"Unrecorded data needs inspection: {target}")
        record.update(status="processing", current=name, pid=os.getpid())
        save_json(receipt, record)
        previous_datasets = dataset_hashes(target) if target.is_file() else {}
        previous_sha = sha(target) if target.is_file() else None
        tick = time.perf_counter()
        deck = decks / f"{name}.njoy"
        print(f"Processing {name} at {desired} K", flush=True)
        with (directory / f"{name}.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            options = dict(njoy_exec=str(njoy), error=specification["njoy_error_tolerance"], stdout=True, input_filename=deck)
            if name in temperatures:
                data = openmc.data.IncidentNeutron.from_njoy(source, temperatures=desired, **options)
            else:
                base = sources / f"{thermal_scattering[name]['nuclide']}.endf"
                data = openmc.data.ThermalScattering.from_njoy(base, source, temperatures=desired, **options)
            if data.name != name:
                raise RuntimeError(f"Processed nuclide name differs: {data.name} versus {name}")
            actual = [float(k / openmc.data.K_BOLTZMANN) for k in data.kTs]
            if len(actual) != len(desired) or any(abs(a-b) > temperature_tolerance
                                                 for a, b in zip(sorted(actual), sorted(desired))):
                raise RuntimeError(f"Processed temperatures differ for {name}: {actual}")
            temporary = target.with_suffix(".h5.tmp")
            data.export_to_hdf5(temporary, "w")
            if previous_datasets:
                current_datasets = dataset_hashes(temporary)
                changed = [key for key, value in previous_datasets.items() if current_datasets.get(key) != value]
                if changed:
                    raise RuntimeError(f"Temperature extension changed existing nuclear-data datasets: {name}: {changed[:8]}")
            temporary.replace(target)
        record["processed"][name] = {"sha256": sha(target), "bytes": target.stat().st_size,
                                    "temperature_K": actual, "wall_s": time.perf_counter() - tick,
                                    "deck": str(deck.relative_to(directory)), "deck_sha256": sha(deck),
                                    "endf_material": evaluation.material}
        if previous_datasets:
            record["processed"][name]["temperature_extension_check"] = {
                "previous_file_sha256": previous_sha, "unchanged_existing_datasets": len(previous_datasets)}
        record["sources"][name]["endf_library"] = list(evaluation.info["library"])
        save_json(receipt, record)
    library = openmc.data.DataLibrary()
    for name in record["processed"]:
        library.register_file(directory / f"{name}.h5")
    library.export_to_xml(directory / "cross_sections.xml")
    record.update(status="prepared", current=None, pid=None,
                  temperature_tolerance_K=temperature_tolerance,
                  python_peak_rss_MiB=peak_rss_mib())
    record.pop("error", None)
    save_json(receipt, record)
    print(f"Prepared {len(record['processed'])} evaluated data files", flush=True)
