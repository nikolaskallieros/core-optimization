"""Download selected evaluated HDF5 tables from an OpenMC library index."""

from pathlib import Path
import json
import shutil
from urllib.parse import urljoin
from urllib.request import urlopen
from xml.etree import ElementTree

import openmc.data

from runtime.io import save_json


def download_library(directory, source_xml, names):
    """Install selected neutron or thermal tables and write a local library index.

    Parameters
    ----------
    directory : str or pathlib.Path
        Persistent data cache. Existing completed tables are reused.
    source_xml : str
        Public URL of an OpenMC cross_sections.xml whose table paths resolve
        relative to that URL.
    names : sequence of str
        Required nuclide or thermal-scattering identities, such as U235 and
        c_H_in_H2O. Only the corresponding files are downloaded.

    Returns
    -------
    pathlib.Path
        Absolute local cross_sections.xml containing all tables retained in this
        cache, including tables installed by an earlier call.

    Raises
    ------
    ValueError
        The cache belongs to another source or a downloaded table's identity
        differs from the requested library entry.

    Notes
    -----
    Download locations and file sizes are retained in manifest.json. A table is
    installed only after OpenMC has read its HDF5 metadata successfully. This
    prepares evaluated data; it does not alter their physics or temperatures.
    """
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "manifest.json"
    record = json.loads(receipt.read_text()) if receipt.exists() else {
        "source_xml": source_xml, "files": {}}
    if record["source_xml"] != source_xml:
        raise ValueError("This nuclear-data cache belongs to a different source")
    with urlopen(source_xml, timeout=60) as response:
        index = ElementTree.fromstring(response.read())
    for name in names:
        entry = next(item.attrib for item in index.findall("library")
                     if name in item.attrib["materials"].split())
        destination = directory / Path(entry["path"]).name
        url = urljoin(source_xml, entry["path"])
        if destination.name not in record["files"]:
            temporary = destination.with_suffix(".part.h5")
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            library = openmc.data.DataLibrary()
            library.register_file(temporary)
            if set(library[0]["materials"]) != set(entry["materials"].split()):
                raise ValueError(f"Downloaded nuclear-data identity differs: {name}")
            temporary.replace(destination)
            record["files"][destination.name] = {
                "url": url, "materials": library[0]["materials"],
                "type": entry["type"], "bytes": destination.stat().st_size}
            save_json(receipt, record)
    library = openmc.data.DataLibrary()
    for filename in record["files"]:
        library.register_file(directory / filename)
    library.export_to_xml(directory / "cross_sections.xml")
    return directory / "cross_sections.xml"
