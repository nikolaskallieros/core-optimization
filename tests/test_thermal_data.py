"""Material-independent thermal-library extension contracts."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import openmc.data
import pytest

from get_nuclear_data import endf
from get_nuclear_data.thermal import extend_thermal_scattering
from runtime.io import save_json, sha


@pytest.fixture
def thermal_library(tmp_path, monkeypatch):
    """Build a small base library with controlled NJOY and download stand-ins.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory supplied by pytest.
    monkeypatch : pytest.MonkeyPatch
        Scoped replacements for native processing and source downloads.

    Returns
    -------
    dict
        Base paths, source choices, processing-call records and failure controls.
        HDF5 files are synthetic fixtures, not evaluated nuclear data.
    """
    base = tmp_path / "base"
    (base / "endf").mkdir(parents=True)
    njoy = tmp_path / "njoy"
    njoy.write_text("test processor")
    calls, control = [], {}
    choices = {
        "c_Graphite": {"nuclide": "C0", "member": "thermal_scatt/tsl-graphite.endf"},
        "c_H_in_H2O": {"nuclide": "H1", "member": "thermal_scatt/tsl-HinH2O.endf"},
        "c_D_in_D2O": {"nuclide": "H2", "member": "thermal_scatt/tsl-DinD2O.endf"},
        "c_H_in_CH2": {"nuclide": "H1", "member": "thermal_scatt/tsl-poly.endf"},
    }
    for name, choice in choices.items():
        choice.update(temperatures_K=[300., 400.],
                      source_sha256=hashlib.sha256(name.encode()).hexdigest())

    def table(name, temperatures, nuclides=()):
        """Construct a deterministic synthetic processed table.

        Parameters
        ----------
        name : str
            Neutron or scattering-table identity.
        temperatures : sequence of float
            Requested table temperatures in K.
        nuclides : sequence of str, optional
            Bound nuclide identities for thermal tables.

        Returns
        -------
        types.SimpleNamespace
            Minimal processed-data object supporting HDF5 export.
        """
        def export(path, mode):
            """Write deterministic temperature datasets for integrity checks.

            Parameters
            ----------
            path : str or pathlib.Path
                Destination HDF5 file.
            mode : str
                HDF5 write mode supplied by the processing interface.
            """
            with h5py.File(path, mode) as handle:
                for temperature in temperatures:
                    value = temperature + (1. if control.get("alter_old") and temperature == 600. else 0.)
                    handle.create_dataset(f"{name}/{temperature:g}K/xs", data=[value, 2. * value])
        actual = np.asarray(temperatures) + (2. if control.get("bad_temperature") and nuclides else 0.)
        return SimpleNamespace(name=name, kTs=actual * openmc.data.K_BOLTZMANN,
                               nuclides=list(nuclides), export_to_hdf5=export)

    source_records, processed = {}, {}
    for name in ("C0", "H1", "H2", "O16"):
        source = base / f"endf/{name}.endf"
        source.write_text(name)
        table(name, [600., 1200.]).export_to_hdf5(base / f"{name}.h5", "w")
        source_records[name] = {"sha256": sha(source)}
        processed[name] = {"sha256": sha(base / f"{name}.h5"), "temperature_K": [600., 1200.]}
    (base / "existing_water.h5").write_bytes(b"existing water table")
    for filename in ("cross_sections.xml", "depletion_cross_sections.xml"):
        library = ET.Element("cross_sections")
        ET.SubElement(library, "directory").text = "."
        for name in processed:
            ET.SubElement(library, "library", materials=name, type="neutron", path=f"{name}.h5")
        ET.SubElement(library, "library", materials="c_H_in_H2O", type="thermal", path="existing_water.h5")
        ET.ElementTree(library).write(base / filename)
    save_json(base / "manifest.json", {
        "library": "ENDF/B-VII.1", "status": "prepared", "njoy_package": None,
        "sources": source_records, "processed": processed,
        "processing_specification": {
            "neutron_temperatures_K": {name: [600., 1200.] for name in processed},
        },
    })
    before = {str(path.relative_to(base)): path.read_bytes() for path in base.rglob("*") if path.is_file()}

    def fetch(record, *, members, sources, receipt):
        """Supply selected synthetic source bytes with their declared checksums.

        Parameters
        ----------
        record : dict
            Mutable processing manifest.
        members : dict
            Selected archive member identities grouped by sublibrary.
        sources : pathlib.Path
            Destination source directory.
        receipt : pathlib.Path
            Processing-manifest output path.
        """
        sources.mkdir(parents=True, exist_ok=True)
        for selected in members.values():
            for name in selected:
                path = sources / f"{name}.endf"
                path.write_text(name)
                assert sha(path) == record["sources"][name]["sha256"]
        save_json(receipt, record)

    def neutron(source, *, temperatures, **options):
        """Record requested neutron temperatures and return a synthetic table.

        Parameters
        ----------
        source : pathlib.Path
            Bound-nuclide neutron evaluation.
        temperatures : sequence of float
            Requested neutron temperatures in K.
        **options : dict
            NJOY options, including the generated input-deck path.

        Returns
        -------
        types.SimpleNamespace
            Deterministic synthetic neutron table.
        """
        name = Path(source).stem
        calls.append(("neutron", name, tuple(temperatures)))
        Path(options["input_filename"]).write_text(name)
        return table(name, temperatures)

    def thermal(neutron_source, source, *, temperatures, **options):
        """Record a thermal-law binding and return its synthetic processed table.

        Parameters
        ----------
        neutron_source : pathlib.Path
            Evaluation for the declared bound nuclide.
        source : pathlib.Path
            Selected thermal-scattering evaluation.
        temperatures : sequence of float
            Requested scattering-table temperatures in K.
        **options : dict
            NJOY options, including the generated input-deck path.

        Returns
        -------
        types.SimpleNamespace
            Synthetic thermal table with controlled temperature and binding errors.
        """
        name, bound = Path(source).stem, Path(neutron_source).stem
        calls.append(("thermal", name, bound, tuple(temperatures)))
        Path(options["input_filename"]).write_text(name)
        return table(name, temperatures, ["unexpected" if control.get("bad_binding") else bound])

    monkeypatch.setattr(endf, "fetch_sources", fetch)
    monkeypatch.setattr(openmc.data.IncidentNeutron, "from_njoy", neutron)
    monkeypatch.setattr(openmc.data.ThermalScattering, "from_njoy", thermal)
    monkeypatch.setattr(openmc.data.endf, "Evaluation", lambda source: SimpleNamespace(
        info={"library": ("ENDF/B", 7, 1)}, material=1))
    return dict(base=base, output=tmp_path / "extension", njoy=njoy, choices=choices,
                calls=calls, control=control, before=before)


@pytest.mark.parametrize("name", ["c_Graphite", "c_H_in_H2O", "c_D_in_D2O"])
def test_extension_uses_declared_material_and_preserves_the_base(thermal_library, name):
    """Verify the same extension interface handles graphite and both water laws.

    Parameters
    ----------
    thermal_library : dict
        Synthetic base-library fixture and processing records.
    name : str
        Scattering-table name under test.
    """
    fixture = thermal_library
    choice = fixture["choices"][name]
    record = extend_thermal_scattering(fixture["base"], fixture["output"], {name: choice},
        error_tolerance=.001, njoy=fixture["njoy"])
    assert record["status"] == "prepared"
    assert record["prior_library_unchanged"]
    assert ("neutron", choice["nuclide"], (300., 400., 600., 1200.)) in fixture["calls"]
    assert ("thermal", name, choice["nuclide"], (300., 400.)) in fixture["calls"]
    for filename in ("cross_sections.xml", "depletion_cross_sections.xml"):
        entries = ET.parse(fixture["output"] / filename).getroot().findall("library")
        selected = [entry for entry in entries if entry.attrib["materials"] == name]
        assert len(selected) == 1
        assert Path(selected[0].attrib["path"]) == fixture["output"] / f"{name}.h5"
        oxygen = next(entry for entry in entries if entry.attrib["materials"] == "O16")
        assert Path(oxygen.attrib["path"]) == fixture["base"] / "O16.h5"
    assert {str(path.relative_to(fixture["base"])): path.read_bytes()
            for path in fixture["base"].rglob("*") if path.is_file()} == fixture["before"]


def test_multiple_scattering_laws_share_one_bound_nuclide_extension(thermal_library):
    """Verify shared bound nuclides are processed once at the temperature union.

    Parameters
    ----------
    thermal_library : dict
        Synthetic base-library fixture and processing records.
    """
    fixture = thermal_library
    selected = {name: dict(fixture["choices"][name]) for name in ("c_H_in_H2O", "c_H_in_CH2")}
    selected["c_H_in_CH2"]["temperatures_K"] = [350., 500.]
    extend_thermal_scattering(fixture["base"], fixture["output"], selected,
                             error_tolerance=.001, njoy=fixture["njoy"])
    assert [call for call in fixture["calls"] if call[0] == "neutron"] == [
        ("neutron", "H1", (300., 350., 400., 500., 600., 1200.))]


def test_verified_extension_reuses_tables_and_supports_transport_only(thermal_library):
    """Verify reuse avoids processing and honors an explicit single XML library.

    Parameters
    ----------
    thermal_library : dict
        Synthetic base-library fixture and processing records.
    """
    fixture = thermal_library
    options = dict(error_tolerance=.001, njoy=fixture["njoy"], library_names=("cross_sections.xml",))
    selected = {"c_H_in_H2O": fixture["choices"]["c_H_in_H2O"]}
    extend_thermal_scattering(fixture["base"], fixture["output"], selected, **options)
    previous = list(fixture["calls"])
    extend_thermal_scattering(fixture["base"], fixture["output"], selected, **options)
    assert fixture["calls"] == previous
    assert not (fixture["output"] / "depletion_cross_sections.xml").exists()


@pytest.mark.parametrize("failure,message", [
    ("alter_old", "changed prior H1 datasets"),
    ("bad_temperature", "temperatures differ"),
    ("bad_binding", "does not bind"),
])
def test_invalid_processed_data_do_not_publish_a_library(thermal_library, failure, message):
    """Verify changed inherited data and invalid thermal tables fail before publication.

    Parameters
    ----------
    thermal_library : dict
        Synthetic base-library fixture and processing records.
    failure : str
        Controlled processing failure enabled for this case.
    message : str
        Expected diagnostic identifying the failed contract.
    """
    fixture = thermal_library
    fixture["control"][failure] = True
    with pytest.raises(RuntimeError, match=message):
        extend_thermal_scattering(fixture["base"], fixture["output"],
            {"c_H_in_H2O": fixture["choices"]["c_H_in_H2O"]},
            error_tolerance=.001, njoy=fixture["njoy"])
    assert not (fixture["output"] / "cross_sections.xml").exists()
    assert json.loads((fixture["output"] / "manifest.json").read_text())["status"] == "failed"
    assert {str(path.relative_to(fixture["base"])): path.read_bytes()
            for path in fixture["base"].rglob("*") if path.is_file()} == fixture["before"]


def test_thermal_extension_cannot_replace_its_base(thermal_library):
    """Reject an extension destination that aliases the inherited data directory.

    Parameters
    ----------
    thermal_library : dict
        Synthetic base-library fixture and processing records.
    """
    fixture = thermal_library
    with pytest.raises(ValueError, match="separate destination"):
        extend_thermal_scattering(fixture["base"], fixture["base"],
            {"c_Graphite": fixture["choices"]["c_Graphite"]},
            error_tolerance=.001, njoy=fixture["njoy"])
