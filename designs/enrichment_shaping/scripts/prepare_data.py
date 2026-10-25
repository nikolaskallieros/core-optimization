"""Install only the evaluated neutron and thermal data used by this design."""

from pathlib import Path
import tomllib

from get_nuclear_data.library import download_library
from runtime.paths import NUCLEAR_DATA

INPUTS = Path(__file__).resolve().parents[1] / "inputs/model.toml"


def main():
    """Download the TOML-selected evaluated data into the shared local cache.

    Returns
    -------
    pathlib.Path
        Local OpenMC cross_sections.xml for this design's native calculations.
    """
    data = tomllib.loads(INPUTS.read_text())["nuclear_data"]
    path = download_library(NUCLEAR_DATA / data["cache"], data["source_xml"], data["names"])
    print(path)
    return path


if __name__ == "__main__":
    main()
