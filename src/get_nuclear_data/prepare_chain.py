"""Unpack a checksum-verified depletion chain; no download is needed."""
import json
import lzma
from pathlib import Path
import shutil
import tempfile

from runtime.io import sha
from runtime.paths import NUCLEAR_CHAINS, NUCLEAR_DATA, NUCLEAR_INPUTS


def prepare_chain(source, destination, record):
    """Unpack the declared depletion chain and verify its file identity.

    Parameters
    ----------
    source : str or pathlib.Path
        Compressed XZ archive containing the chain XML.
    destination : str or pathlib.Path
        Destination XML path; an existing different chain is rejected.
    record : dict
        Manifest entry with compressed_sha256 and sha256 fingerprints for the archive and
        unpacked XML.

    Returns
    -------
    pathlib.Path
        Verified destination XML path.

    Raises
    ------
    RuntimeError
        The archive, unpacked XML or existing destination differs from its declared
        fingerprint.

    Notes
    -----
    Checksums establish byte identity with the declared data version. They do not validate
    the physical content of the depletion chain.
    """
    source, destination = Path(source), Path(destination)
    if sha(source) != record["compressed_sha256"]:
        raise RuntimeError("Bundled depletion-chain archive failed verification")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                             suffix=".partial", delete=False) as outgoing:
                temporary = Path(outgoing.name)
                with lzma.open(source, "rb") as incoming:
                    shutil.copyfileobj(incoming, outgoing)
            if sha(temporary) != record["sha256"]:
                raise RuntimeError("Unpacked depletion chain failed verification")
            temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    if sha(destination) != record["sha256"]:
        raise RuntimeError("Existing depletion chain differs from the bundled source")
    return destination


def main():
    """Prepare the bundled depletion chain and print its declared nuclide count."""
    record = json.loads((NUCLEAR_INPUTS / "depletion_data.json").read_text())["chain"]
    prepare_chain(NUCLEAR_CHAINS / "chain_endfb71_pwr.xml.xz",
                  NUCLEAR_DATA / "chains/chain_endfb71_pwr.xml", record)
    print(f"Verified depletion chain: {record['nuclides']} nuclides.")


if __name__ == "__main__":
    main()
