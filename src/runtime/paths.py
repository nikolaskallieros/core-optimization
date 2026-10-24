"""Package resources and caller-selected workspace locations for data and runs."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("CORE_OPTIMIZATION_ROOT", Path.cwd())).expanduser().resolve()
NUCLEAR_PACKAGE = Path(__file__).resolve().parents[1] / "get_nuclear_data"
NUCLEAR_INPUTS = NUCLEAR_PACKAGE / "manifests"
NUCLEAR_CHAINS = NUCLEAR_PACKAGE / "chains"
NUCLEAR_DATA = Path(os.environ.get("CORE_OPTIMIZATION_DATA", PROJECT_ROOT / ".cache/nuclear_data")).expanduser().resolve()
RUNS = PROJECT_ROOT / "runs"
