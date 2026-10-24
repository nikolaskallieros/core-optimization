"""Extend evaluated libraries with explicitly selected thermal-scattering tables."""
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

from get_nuclear_data import endf
from get_nuclear_data.processing import processing_context
from get_nuclear_data.verification import dataset_hashes
from runtime.io import sha, save_json


def extend_thermal_scattering(base, directory, thermal_scattering, *, error_tolerance,
                              njoy=None, temperature_tolerance=.05,
                              library_names=("cross_sections.xml", "depletion_cross_sections.xml")):
    """Extend a verified library with caller-selected scattering materials.

    Parameters
    ----------
    base : str or pathlib.Path
        Prepared ENDF/B-VII.1 library containing its manifest, neutron HDF5
        tables and source neutron evaluations under ``endf/``.
    directory : str or pathlib.Path
        Separate destination for extended tables, library XML and processing
        records. Inherited files remain at their original paths.
    thermal_scattering : dict
        Mapping from OpenMC scattering-table name to a specification containing
        ``nuclide``, archive ``member``, ``temperatures_K`` and ``source_sha256``.
        For example, c_H_in_H2O binds to H1 and c_Graphite binds to C0.
    error_tolerance : float
        Relative NJOY reconstruction tolerance.
    njoy : str or pathlib.Path, optional
        NJOY executable; None uses ``bin/njoy`` in the active environment.
    temperature_tolerance : float, optional
        Allowed difference in K between requested and processed temperatures.
    library_names : sequence of str, optional
        Base XML libraries to extend. Each named file must exist. Supply only
        ``cross_sections.xml`` for a transport-only library.

    Returns
    -------
    dict
        Processing record containing selected sources, generated-file checksums,
        temperatures and verification that inherited data stayed unchanged.

    Raises
    ------
    ValueError
        The destination is the base directory or no scattering tables are selected.
    RuntimeError
        Existing data conflict with the declared processing identity, or processed
        names, bound nuclides, temperatures or inherited datasets differ.

    Notes
    -----
    Each bound nuclide is processed once at the union of its inherited and requested
    temperatures. Multiple scattering laws can share the same bound nuclide.
    Existing thermal entries with matching names are replaced in the new XML;
    unrelated neutron and thermal entries continue to reference the base files.
    """
    base, directory = Path(base).resolve(), Path(directory).resolve()
    if base == directory or not thermal_scattering:
        raise ValueError("A thermal extension needs selected tables and a separate destination")
    njoy = Path(njoy) if njoy is not None else Path(sys.prefix) / "bin/njoy"
    receipt = directory / "manifest.json"
    with processing_context(directory, receipt):
        return _prepare(base, directory, thermal_scattering, error_tolerance,
                        njoy, temperature_tolerance, tuple(library_names), receipt)


def _prepare(base, directory, thermal_scattering, error_tolerance, njoy,
             temperature_tolerance, library_names, receipt):
    """Process selected thermal tables inside the managed scratch context.

    Parameters
    ----------
    base : pathlib.Path
        Verified parent library directory.
    directory : pathlib.Path
        Separate extension directory.
    thermal_scattering : dict
        Scattering-table names and their explicit source, bound nuclide and
        temperature specifications.
    error_tolerance : float
        Relative NJOY reconstruction tolerance.
    njoy : pathlib.Path
        NJOY executable used to process the requested temperature tables.
    temperature_tolerance : float
        Allowed processed-temperature discrepancy in K.
    library_names : tuple of str
        Base XML filenames to extend.
    receipt : pathlib.Path
        Destination processing manifest, updated after each completed table.

    Returns
    -------
    dict
        Completed processing manifest.
    """
    import openmc.data

    decks = directory / "decks"
    inherited = json.loads((base / "manifest.json").read_text())
    neutron_temperatures = {}
    inherited_files = {name: sha(base / name) for name in library_names}
    for choice in thermal_scattering.values():
        nuclide = choice["nuclide"]
        temperatures = neutron_temperatures.setdefault(
            nuclide, set(inherited["processing_specification"]["neutron_temperatures_K"][nuclide]))
        temperatures.update(choice["temperatures_K"])
        for name, expected in (
            (f"{nuclide}.h5", inherited["processed"][nuclide]["sha256"]),
            (f"endf/{nuclide}.endf", inherited["sources"][nuclide]["sha256"]),
        ):
            if sha(base / name) != expected:
                raise RuntimeError(f"Inherited neutron data failed verification: {name}")
            inherited_files[name] = expected
    neutron_temperatures = {name: sorted(values) for name, values in neutron_temperatures.items()}
    identity = {
        "library": "ENDF/B-VII.1", "njoy_package": inherited["njoy_package"],
        "base_files_sha256": inherited_files, "thermal_scattering": thermal_scattering,
        "error_tolerance": error_tolerance, "neutron_temperatures_K": neutron_temperatures,
        "temperature_tolerance_K": temperature_tolerance,
    }
    record = json.loads(receipt.read_text()) if receipt.exists() else {
        **identity, "sources": {
            **{name: inherited["sources"][name] for name in neutron_temperatures},
            **{name: {"sha256": choice["source_sha256"]} for name, choice in thermal_scattering.items()},
        }, "processed": {}, "openmc_version": openmc.__version__,
    }
    if any(record.get(key) != value for key, value in identity.items()):
        raise RuntimeError("Existing thermal extension has a different processing identity")
    decks.mkdir(exist_ok=True)
    record.update(status="downloading", pid=os.getpid())
    save_json(receipt, record)
    endf.fetch_sources(record, members={"thermal_scatt": {
        name: choice["member"] for name, choice in thermal_scattering.items()}},
        sources=directory / "endf", receipt=receipt)

    tables = {**neutron_temperatures,
              **{name: choice["temperatures_K"] for name, choice in thermal_scattering.items()}}
    for name, temperatures in tables.items():
        target = directory / f"{name}.h5"
        if name in record["processed"]:
            if sha(target) != record["processed"][name]["sha256"]:
                raise RuntimeError(f"Processed thermal-extension data failed verification: {name}")
            continue
        if target.exists():
            raise RuntimeError(f"Unrecorded data requires inspection: {target}")
        record.update(status="processing", current=name)
        save_json(receipt, record)
        started = time.perf_counter()
        deck = decks / f"{name}.njoy"
        options = dict(njoy_exec=str(njoy), error=error_tolerance,
                       stdout=True, input_filename=deck)
        with (directory / f"{name}.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if name in neutron_temperatures:
                data = openmc.data.IncidentNeutron.from_njoy(
                    base / f"endf/{name}.endf", temperatures=temperatures, **options)
            else:
                choice = thermal_scattering[name]
                source = directory / f"endf/{name}.endf"
                evaluation = openmc.data.endf.Evaluation(source)
                if tuple(evaluation.info["library"]) != ("ENDF/B", 7, 1):
                    raise RuntimeError(f"Unexpected evaluated library for {name}")
                data = openmc.data.ThermalScattering.from_njoy(
                    base / f"endf/{choice['nuclide']}.endf", source,
                    temperatures=temperatures, **options)
                if choice["nuclide"] not in data.nuclides:
                    raise RuntimeError(f"{name} does not bind to the declared nuclide {choice['nuclide']}")
                record["sources"][name].update(endf_material=evaluation.material,
                                               endf_library=list(evaluation.info["library"]))
            actual = [float(k / openmc.data.K_BOLTZMANN) for k in data.kTs]
            if (data.name != name or len(actual) != len(temperatures)
                    or any(abs(a - b) > temperature_tolerance
                           for a, b in zip(sorted(actual), sorted(temperatures)))):
                raise RuntimeError(f"Processed name or temperatures differ for {name}: {actual}")
            temporary = target.with_suffix(".h5.tmp")
            data.export_to_hdf5(temporary, "w")
            if name in neutron_temperatures:
                original = dataset_hashes(base / f"{name}.h5")
                updated = dataset_hashes(temporary)
                changed = [key for key, value in original.items() if updated.get(key) != value]
                if changed:
                    raise RuntimeError(f"Temperature extension changed prior {name} datasets: {changed[:8]}")
            temporary.replace(target)
        record["processed"][name] = {
            "sha256": sha(target), "bytes": target.stat().st_size, "temperature_K": actual,
            "wall_s": time.perf_counter() - started, "deck": str(deck.relative_to(directory)),
            "deck_sha256": sha(deck),
        }
        if name in neutron_temperatures:
            record["processed"][name]["unchanged_existing_datasets"] = len(original)
        else:
            record["processed"][name]["bound_nuclides"] = list(data.nuclides)
        save_json(receipt, record)

    for name in library_names:
        original = ET.parse(base / name).getroot()
        library = ET.Element("cross_sections")
        inherited_directory = Path(original.findtext("directory", str(base)))
        if not inherited_directory.is_absolute():
            inherited_directory = base / inherited_directory
        for entry in original.findall("library"):
            attributes = dict(entry.attrib)
            material = attributes["materials"]
            if attributes["type"] == "thermal" and material in thermal_scattering:
                continue
            path = Path(attributes["path"])
            if not path.is_absolute():
                path = inherited_directory / path
            if attributes["type"] == "neutron" and material in neutron_temperatures:
                path = directory / f"{material}.h5"
            attributes["path"] = str(path.resolve())
            ET.SubElement(library, "library", attributes)
        for material in thermal_scattering:
            ET.SubElement(library, "library", materials=material, type="thermal",
                          path=str(directory / f"{material}.h5"))
        ET.indent(library)
        ET.ElementTree(library).write(directory / name, encoding="utf-8", xml_declaration=True)
    if any(sha(base / name) != digest for name, digest in inherited_files.items()):
        raise RuntimeError("Preparing thermal scattering altered inherited nuclear data")
    record.update(status="prepared", current=None, pid=None, prior_library_unchanged=True,
                  library_files_sha256={name: sha(directory / name) for name in library_names})
    record.pop("error", None)
    save_json(receipt, record)
    return record
