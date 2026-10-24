"""Shared data preparation contracts, independent of a reactor example."""
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np
import openmc.data
import pytest

from get_nuclear_data.prepare_chain import prepare_chain
from get_nuclear_data.processing import processing_context
from get_nuclear_data.verification import dataset_hashes, table_collapse
from runtime.io import HashStream, save_json, sha


def test_json_rejects_nonfinite_values_without_losing_the_previous_record(tmp_path):
    """Verify JSON rejects nonfinite values without losing the previous record.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    path = tmp_path / "nested/record.json"
    save_json(path, {"values": np.array([1., 2.]), "count": np.int64(2)})
    assert json.loads(path.read_text()) == {"values": [1., 2.], "count": 2}
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="Out of range"):
        save_json(path, {"values": np.array([np.nan])})
    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]


def test_json_replace_failure_preserves_record_and_cleans_temporary_file(tmp_path, monkeypatch):
    """Verify JSON replace failure preserves record and cleans temporary file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacements for dependencies or environment settings.
    """
    path = tmp_path / "record.json"
    save_json(path, {"generation": 1})

    def fail_replace(self, target):
        """Inject an operating-system failure at the atomic replacement.

        Parameters
        ----------
        target : str or pathlib.Path
            Requested replacement destination.

        Raises
        ------
        OSError
            Always raised to test failure cleanup.
        """
        raise OSError("replacement unavailable")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement unavailable"):
        save_json(path, {"generation": 2})
    assert json.loads(path.read_text()) == {"generation": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_hash_stream_supports_buffered_archive_consumers():
    """Verify hash stream supports buffered archive consumers."""
    payload = b"evaluated table\n" * 100
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w") as archive:
        entry = tarfile.TarInfo("table.endf")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    data = memory.getvalue()
    progress = []
    stream = HashStream(io.BytesIO(data), progress=progress.append, notify_every=512)
    with io.BufferedReader(stream, buffer_size=512) as buffered:
        with tarfile.open(fileobj=buffered, mode="r|*") as archive:
            assert archive.extractfile(next(iter(archive))).read() == payload
        buffered.read()
    assert stream.count == len(data)
    assert stream.digest.hexdigest() == hashlib.sha256(data).hexdigest()
    assert progress and progress == sorted(set(progress))


@pytest.fixture
def chain_files(tmp_path):
    """Write a tiny compressed chain with known byte identities.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.

    Returns
    -------
    tuple
        Compressed source path, intended output path, declared digests and original XML
        bytes.
    """
    payload = b'<depletion_chain><nuclide name="U235"/></depletion_chain>\n'
    source = tmp_path / "chain.xml.xz"
    source.write_bytes(lzma.compress(payload))
    record = {"compressed_sha256": sha(source), "sha256": hashlib.sha256(payload).hexdigest()}
    return source, tmp_path / "prepared/chain.xml", record, payload


def test_chain_preparation_verifies_existing_files_without_overwriting_them(chain_files):
    """Verify chain preparation verifies existing files without overwriting them.

    Parameters
    ----------
    chain_files : tuple
        Synthetic chain archive, destination, checksum record and expected XML bytes.
    """
    source, destination, record, payload = chain_files
    assert prepare_chain(source, destination, record) == destination
    assert destination.read_bytes() == payload
    assert prepare_chain(source, destination, record) == destination
    destination.write_bytes(b"different chain")
    with pytest.raises(RuntimeError, match="Existing depletion chain differs"):
        prepare_chain(source, destination, record)
    assert destination.read_bytes() == b"different chain"


@pytest.mark.parametrize("field", ["compressed_sha256", "sha256"])
def test_chain_hash_failure_does_not_publish_a_partial_chain(chain_files, field):
    """Verify chain hash failure does not publish a partial chain.

    Parameters
    ----------
    chain_files : tuple
        Synthetic chain archive, destination, checksum record and expected XML bytes.
    field : str
        Archive or unpacked-chain checksum field deliberately made inconsistent.
    """
    source, destination, record, _ = chain_files
    with pytest.raises(RuntimeError, match="failed verification"):
        prepare_chain(source, destination, {**record, field: "0" * 64})
    assert not destination.exists()
    assert not destination.parent.exists() or not list(destination.parent.iterdir())


def test_temperature_addition_preserves_existing_dataset_fingerprints(tmp_path):
    """Verify temperature addition preserves existing dataset fingerprints.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    path = tmp_path / "tables.h5"
    with h5py.File(path, "w") as handle:
        table = handle.create_dataset("300K/capture", data=[1., 2., 3.])
        table.attrs["units"] = "barn"
    initial = dataset_hashes(path)
    with h5py.File(path, "a") as handle:
        handle.create_dataset("600K/capture", data=[2., 3., 4.])
    extended = dataset_hashes(path)
    assert extended["300K/capture"] == initial["300K/capture"]
    assert set(extended) == {"300K/capture", "600K/capture"}
    with h5py.File(path, "a") as handle:
        handle["300K/capture"].attrs["units"] = "cm2"
    assert dataset_hashes(path)["300K/capture"] != initial["300K/capture"]
    with h5py.File(path, "a") as handle:
        handle["600K/capture"][1] = 99.
    assert dataset_hashes(path)["600K/capture"] != extended["600K/capture"]


@pytest.mark.parametrize("temperature,mt,expected", [(450., 102, 28.625), (300., 102, 16.75), (600., 18, 14.)])
def test_evaluated_rate_collapse_uses_explicit_reaction_and_temperature(monkeypatch, temperature, mt, expected):
    """Verify evaluated rate collapse uses explicit reaction and temperature.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped replacements for dependencies or environment settings.
    temperature : float
        Requested temperature in K for the independent reaction-table collapse.
    mt : int
        ENDF reaction MT identifier selected in the synthetic table.
    expected : float
        Analytical collapsed reaction score for the selected synthetic table and
        temperature.
    """
    energy = np.array([0., 1., 4., 10.])
    table = openmc.data.Tabulated1D
    data = SimpleNamespace(
        temperatures=["600K", "300K"],
        kTs=np.array([600., 300.]) * openmc.data.K_BOLTZMANN,
        reactions={
            102: SimpleNamespace(xs={"300K": table(energy, 1. + .5 * energy),
                                     "600K": table(energy, 3. + energy)}),
            18: SimpleNamespace(xs={"300K": table(energy, np.ones(4)),
                                    "600K": table(energy, 2. * np.ones(4))}),
        },
    )
    monkeypatch.setattr(openmc.data.IncidentNeutron, "from_hdf5", lambda path: data)
    value = table_collapse("fixture.h5", np.array([0., 2., 5.]), np.array([2., 5.]),
                           temperature=temperature, mt=mt)
    assert value == pytest.approx(expected)


def test_processing_failure_restores_environment_and_retains_its_error(tmp_path, monkeypatch):
    """Verify processing failure restores environment and retains its error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacements for dependencies or environment settings.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    previous_temp = tempfile.tempdir
    receipt = tmp_path / "manifest.json"
    save_json(receipt, {"status": "processing", "pid": 123})
    with pytest.raises(RuntimeError, match="processing failed"):
        with processing_context(tmp_path, receipt):
            assert Path(tempfile.tempdir) == tmp_path / "scratch"
            assert os.environ["OMP_NUM_THREADS"] == "1"
            raise RuntimeError("processing failed")
    assert tempfile.tempdir == previous_temp
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert json.loads(receipt.read_text()) == {
        "status": "failed", "pid": None, "error": "RuntimeError: processing failed"}


def test_depletion_preparation_download_and_reuse_preserve_the_base_directory(tmp_path, monkeypatch):
    """Verify depletion preparation download and reuse preserve the base directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacements for dependencies or environment settings.
    """
    from get_nuclear_data import depletion

    directory = tmp_path / "library"
    directory.mkdir()
    temperature = 900.
    sources = {}
    for name in ("H1", "He4"):
        path = tmp_path / f"{name}.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["filetype"] = np.bytes_("data_neutron")
            handle.create_dataset(f"{name}/kTs/900K", data=temperature * openmc.data.K_BOLTZMANN)
        sources[name] = path.read_bytes()
    (directory / "H1.h5").write_bytes(sources["H1"])
    save_json(directory / "manifest.json", {
        "status": "prepared", "processing_specification": {"neutron_temperatures_K": {"H1": [temperature]}},
        "processed": {"H1": {"sha256": sha(directory / "H1.h5"), "temperature_K": [temperature]}}})
    chain = b'<depletion_chain><nuclide name="H1" reactions="0"/><nuclide name="He4" reactions="0"/></depletion_chain>'
    compressed = tmp_path / "custom_chain.xml.xz"
    compressed.write_bytes(lzma.compress(chain))
    selection = tmp_path / "selection.json"
    save_json(selection, {"transport_data_files": [{"name": name} for name in sources],
                         "chain": {"sha256": hashlib.sha256(chain).hexdigest(),
                                   "compressed_sha256": sha(compressed), "nuclides": 2}})
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w:xz") as archive:
        member = tarfile.TarInfo("selected/neutron/He4.h5")
        member.size = len(sources["He4"])
        archive.addfile(member, io.BytesIO(sources["He4"]))
    payload = memory.getvalue()
    expected = tmp_path / "expected.json"
    save_json(expected, {"archive_sha256": hashlib.sha256(payload).hexdigest(),
                         "files": {"He4": {"sha256": hashlib.sha256(sources["He4"]).hexdigest()}}})
    monkeypatch.setattr(depletion, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    kwargs = dict(archive_url="https://example.invalid/data.xz", archive_bytes=len(payload),
                  source_page="https://example.invalid/", fuel_temperature_K=temperature)
    depletion.prepare_depletion_data(directory, selection, expected, compressed, **kwargs)
    xml = directory / "depletion_cross_sections.xml"
    library = openmc.data.DataLibrary.from_xml(xml)
    assert {Path(item["path"]) for item in library} == {
        directory / "H1.h5", directory / "depletion_neutron/He4.h5"}
    assert (directory / "H1.h5").read_bytes() == sources["H1"]
    assert (directory / "custom_chain.xml").read_bytes() == chain
    first_xml = xml.read_bytes()
    monkeypatch.setattr(depletion, "urlopen", lambda *args, **kwargs: pytest.fail("Verified data must be reused"))
    depletion.prepare_depletion_data(directory, selection, expected, compressed, **kwargs)
    assert xml.read_bytes() == first_xml
    assert json.loads((directory / "depletion_manifest.json").read_text())["status"] == "prepared"
