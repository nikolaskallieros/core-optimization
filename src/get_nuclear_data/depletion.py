"""Install selected, hash-verified official neutron data and a full depletion chain."""
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
from urllib.request import urlopen
from runtime.io import sha, save_json, HashStream
from get_nuclear_data.prepare_chain import prepare_chain
from get_nuclear_data.processing import processing_context


def prepare_depletion_data(directory, selection_manifest, source_manifest, compressed, *,
                           archive_url, archive_bytes, source_page, fuel_temperature_K):
    """Install the declared depletion chain and required neutron evaluations.

    Parameters
    ----------
    directory : str or pathlib.Path
        Destination nuclear-data cache for tables, NJOY decks, logs and processing receipts.
    selection_manifest : str or pathlib.Path
        JSON manifest declaring the full chain and required neutron evaluations.
    source_manifest : str or pathlib.Path
        JSON source manifest containing expected evaluation checksums.
    compressed : str or pathlib.Path
        Bundled compressed depletion-chain XML archive.
    archive_url : str
        URL of the official preprocessed neutron-data archive.
    archive_bytes : int
        Declared archive byte count used for integrity checks and progress.
    source_page : str
        Source publication URL retained in the processing receipt.
    fuel_temperature_K : float
        Required fuel temperature in K for every selected neutron evaluation.

    Raises
    ------
    RuntimeError
        Archive content, existing files, required temperatures or chain size differ from the
        declared inputs.

    Notes
    -----
    Completed downloads and processing metadata are retained in the destination; conflicting
    existing data fail verification instead of being replaced.
    """
    directory, source_manifest = Path(directory), Path(source_manifest)
    receipt = directory / "depletion_manifest.json"
    with processing_context(directory, receipt):
        return _prepare(directory, selection_manifest, source_manifest, compressed,
                        archive_url, archive_bytes, source_page, fuel_temperature_K, receipt)


def _prepare(directory, selection_manifest, source_manifest, compressed,
             archive_url, archive_bytes, source_page, fuel_temperature_K, receipt):
    """Install verified depletion data and assemble their OpenMC library XML.

    Parameters
    ----------
    directory : str or pathlib.Path
        Destination nuclear-data cache for tables, NJOY decks, logs and processing receipts.
    selection_manifest : str or pathlib.Path
        JSON manifest declaring the full chain and required neutron evaluations.
    source_manifest : str or pathlib.Path
        JSON source manifest containing expected evaluation checksums.
    compressed : str or pathlib.Path
        Bundled compressed depletion-chain XML archive.
    archive_url : str
        URL of the official preprocessed neutron-data archive.
    archive_bytes : int
        Declared archive byte count used for integrity checks and progress.
    source_page : str
        Source publication URL retained in the processing receipt.
    fuel_temperature_K : float
        Required fuel temperature in K for every selected neutron evaluation.
    receipt : pathlib.Path
        Processing-manifest destination updated after completed stages.

    Notes
    -----
    Streams selected neutron files from the release archive, verifies the full archive, and
    preserves the base material library. If base preparation is incomplete, retains the
    download and defers final library assembly.
    """
    base_receipt = directory / "manifest.json"
    import openmc.data
    import openmc.deplete

    selected = json.loads(Path(selection_manifest).read_text())
    expected = json.loads(source_manifest.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    chain = directory / Path(compressed).with_suffix("").name
    prepare_chain(compressed, chain, selected["chain"])
    base = json.loads(base_receipt.read_text())
    base_names = set(base["processing_specification"]["neutron_temperatures_K"])
    needed = {item["name"] for item in selected["transport_data_files"]} - base_names
    record = json.loads(receipt.read_text()) if receipt.exists() else {
        "status": "prepared_download", "source_page": source_page, "archive_url": archive_url,
        "library": "ENDF/B-VII.1", "official_processing": "NJOY 2016.68; published six-temperature neutron library",
        "chain": selected["chain"], "files": {}, "archive_complete": False}
    if not record["archive_complete"]:
        neutron_directory = directory / "depletion_neutron"
        neutron_directory.mkdir(exist_ok=True)
        record.update(status="downloading", pid=os.getpid())
        save_json(receipt, record)
        started = time.perf_counter()
        with urlopen(archive_url, timeout=60) as response:
            stream = HashStream(response, progress=lambda count: print(
                f"Official neutron data: {count / archive_bytes:.0%} of archive read", flush=True))
            with tarfile.open(fileobj=stream, mode="r|xz") as archive:
                for member in archive:
                    path = PurePosixPath(member.name)
                    name = path.stem
                    if not member.isfile() or path.suffix != ".h5" or name not in needed:
                        continue
                    if "neutron" not in path.parts:
                        continue
                    destination = neutron_directory / path.name
                    temporary = destination.with_suffix(".download")
                    with archive.extractfile(member) as source, temporary.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    digest = sha(temporary)
                    if digest != expected["files"][name]["sha256"]:
                        raise RuntimeError(f"Official neutron data differs from the retained source: {name}")
                    previous = record["files"].get(name)
                    if previous is not None and previous["sha256"] != digest:
                        raise RuntimeError(f"Official archive member changed: {name}")
                    if destination.exists():
                        if sha(destination) != digest:
                            raise RuntimeError(f"Existing nuclear data conflicts: {name}")
                        temporary.unlink()
                    else:
                        temporary.rename(destination)
                    record["files"][name] = {"member": member.name, "bytes": member.size, "sha256": digest}
                    if len(record["files"]) % 20 == 0:
                        save_json(receipt, record)
            while stream.read(1024 * 1024):
                pass
            if stream.count != archive_bytes:
                raise RuntimeError(f"Official archive size changed: {stream.count}")
            if stream.digest.hexdigest() != expected["archive_sha256"]:
                raise RuntimeError("Official neutron archive checksum mismatch")
            record.update(archive_sha256=stream.digest.hexdigest(), archive_bytes=stream.count,
                          download_wall_s=time.perf_counter() - started, archive_complete=True,
                          status="downloaded", pid=None)
            save_json(receipt, record)
    missing = needed - set(record["files"])
    if missing:
        raise RuntimeError(f"Missing required neutron evaluations: {sorted(missing)}")
    if base["status"] != "prepared":
        print("Official depletion data downloaded; finalize after the base material data is prepared.", flush=True)
        return
    library = openmc.data.DataLibrary()
    for name, item in base["processed"].items():
        path = directory / f"{name}.h5"
        if sha(path) != item["sha256"]:
            raise RuntimeError(f"Base material data changed: {name}")
        if name in base_names and min(abs(t - fuel_temperature_K) for t in item["temperature_K"]) > .1:
            raise RuntimeError(f"Fuel-temperature depletion data are missing for base nuclide {name}")
        library.register_file(path)
    import h5py
    for name, item in record["files"].items():
        path = directory / "depletion_neutron" / f"{name}.h5"
        if sha(path) != item["sha256"]:
            raise RuntimeError(f"Official neutron data changed: {name}")
        with h5py.File(path) as handle:
            data = handle[name]
            temperatures = [float(t[()]) / openmc.data.K_BOLTZMANN for t in data["kTs"].values()]
            if min(abs(t - fuel_temperature_K) for t in temperatures) > .1:
                raise RuntimeError(f"Missing {fuel_temperature_K} K data for {name}")
            item["temperature_K"] = temperatures
        library.register_file(path)
    library.export_to_xml(directory / "depletion_cross_sections.xml")
    parsed = openmc.deplete.Chain.from_xml(chain)
    if len(parsed.nuclides) != selected["chain"]["nuclides"]:
        raise RuntimeError("The full depletion chain has a different nuclide count")
    record.update(status="prepared", pid=None, neutron_evaluations=len(needed | base_names),
                  chain_nuclides=len(parsed.nuclides), chain_sha256=sha(chain))
    record.pop("error", None)
    save_json(receipt, record)
    print(f"Full {len(parsed.nuclides)}-nuclide chain and hot neutron library prepared", flush=True)
