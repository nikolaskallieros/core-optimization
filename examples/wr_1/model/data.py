"""Evaluated-data choices for WR-1; processing is supplied by get_nuclear_data."""
from pathlib import Path
import shutil
import sys
from get_nuclear_data.neutron import prepare_neutron_data
from get_nuclear_data.depletion import prepare_depletion_data
from get_nuclear_data.thermal import extend_thermal_scattering
from coupling.cases import load_configuration
from runtime.paths import NUCLEAR_DATA, NUCLEAR_CHAINS, NUCLEAR_INPUTS
from examples.wr_1 import WR1_ROOT

MINIMUM_FREE_BYTES = 10 * 1024**3


def prepare():
    """Prepare the evaluated data selected by the authoritative study inputs.

    Raises
    ------
    OSError
        Less than 10 GiB is available for evaluated data and processing scratch files.
    FileNotFoundError
        The active environment does not contain NJOY.

    Notes
    -----
    Loads the model and source manifests through study.toml, then prepares neutron,
    depletion and graphite libraries in the configured nuclear-data cache.
    """
    configuration = load_configuration(WR1_ROOT / "inputs/study.toml", 1)
    transport = configuration.data("model")["transport"]
    spec = transport["nuclear_data"]
    depletion = configuration.data("depletion_sources")
    NUCLEAR_DATA.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(NUCLEAR_DATA).free
    if free < MINIMUM_FREE_BYTES:
        raise OSError(f"Data and scratch need 10 GiB free; {free / 1024**3:.1f} GiB available at {NUCLEAR_DATA}")
    njoy = Path(sys.prefix) / "bin/njoy"
    if not njoy.is_file():
        raise FileNotFoundError("NJOY is missing; create the environment from environment.yml")
    base = NUCLEAR_DATA / "wr_1"
    prepare_neutron_data(base, spec, configuration.files["neutron_sources"],
                         thermal_scattering=spec["thermal_scattering"], njoy=njoy)
    prepare_depletion_data(base, NUCLEAR_INPUTS / "depletion_data.json",
        configuration.files["depletion_sources"], NUCLEAR_CHAINS / "chain_endfb71_pwr.xml.xz",
        archive_url=depletion["archive_url"], archive_bytes=depletion["archive_bytes"],
        source_page=depletion["source_page"], fuel_temperature_K=transport["homogenized_element_temperature_K"])
    for name, tables in spec["extensions"].items():
        extend_thermal_scattering(base, NUCLEAR_DATA / name, tables,
            error_tolerance=spec["njoy_error_tolerance"], njoy=njoy)
