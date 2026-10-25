"""Keep the installed library usable without an example checkout."""
import os
import subprocess
import sys


def test_shared_packages_import_from_an_independent_workspace(tmp_path):
    """Verify shared packages import from an independent workspace.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory supplied by pytest.
    """
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("CORE_OPTIMIZATION_ROOT", "CORE_OPTIMIZATION_DATA", "PYTHONPATH")}
    script = """
import importlib
from pathlib import Path
import pkgutil
import sys
for name in ('coupling', 'depletion', 'diffusion_surrogate', 'fuel_cycle',
             'geometry', 'get_nuclear_data', 'materials', 'optimization', 'plotting', 'runtime'):
    package = importlib.import_module(name)
    for module in pkgutil.walk_packages(package.__path__, prefix=name + '.'):
        importlib.import_module(module.name)
    for exported in getattr(package, '__all__', ()):
        getattr(package, exported)
assert not any(name == 'examples' or name.startswith('examples.') for name in sys.modules)
assert importlib.util.find_spec('infrastructure') is None
from runtime.paths import PROJECT_ROOT, NUCLEAR_DATA, NUCLEAR_CHAINS
assert PROJECT_ROOT == Path.cwd()
assert NUCLEAR_DATA == Path.cwd() / '.cache/nuclear_data'
assert (NUCLEAR_CHAINS / 'chain_endfb71_pwr.xml.xz').is_file()
from importlib.resources import files
assert files('depletion.kernels').joinpath('LICENSE').is_file()
"""
    subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path, env=environment,
                   capture_output=True, text=True, check=True)
