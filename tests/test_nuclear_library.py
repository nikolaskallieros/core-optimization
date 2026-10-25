"""Check selective download and material identity without network access."""

import json

import h5py
import numpy as np
import pytest

from get_nuclear_data.library import download_library


def test_selected_library_download_reuses_only_the_declared_tables(tmp_path):
    """Unrequested data remain unfetched and the local index keeps table identities.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated source and cache directories.
    """
    source = tmp_path / "source"
    source.mkdir()
    with h5py.File(source / "H1.h5", "w") as output:
        output.attrs["filetype"] = np.bytes_("data_neutron")
        output.create_group("H1")
    index = source / "cross_sections.xml"
    index.write_text('<cross_sections><library materials="H1" path="H1.h5" type="neutron"/>'
                     '<library materials="U235" path="absent.h5" type="neutron"/></cross_sections>')
    destination = tmp_path / "cache"
    local = download_library(destination, index.as_uri(), ["H1"])
    assert local.is_file()
    assert set(json.loads((destination / "manifest.json").read_text())["files"]) == {"H1.h5"}
    (source / "H1.h5").unlink()
    assert download_library(destination, index.as_uri(), ["H1"]) == local


def test_wrong_nuclide_cannot_be_installed_under_an_expected_name(tmp_path):
    """Reject a source-index/table identity disagreement before native execution.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated source and cache paths.
    """
    with h5py.File(tmp_path / "wrong.h5", "w") as output:
        output.attrs["filetype"] = np.bytes_("data_neutron")
        output.create_group("H1")
    index = tmp_path / "cross_sections.xml"
    index.write_text('<cross_sections><library materials="U235" path="wrong.h5" type="neutron"/></cross_sections>')
    with pytest.raises(ValueError, match="identity differs"):
        download_library(tmp_path / "cache", index.as_uri(), ["U235"])
    assert not (tmp_path / "cache/wrong.h5").exists()
