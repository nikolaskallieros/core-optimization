"""Check the installed packages without changing retained scientific results."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation-output"
DIFFUSION = ROOT / "tests/diffusion"
CHECKS = ("check_solver.py", "check_solver3d.py", "check_fitting.py", "check_fitting3d.py",
          "check_interfaces.py", "check_design_loop.py", "check_equilibrium_convergence.py")


def run(command, name, environment):
    """Execute one installation check and retain its combined log.

    Parameters
    ----------
    command : sequence of str
        Subprocess executable and argument vector.
    name : str
        Check label used for the log filename and completion message.
    environment : mapping of str to str
        Complete environment passed to the subprocess.
    """
    with (OUTPUT / f"{name}.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Passed: {name}", flush=True)


def main():
    """Run bounded installation checks and save their completion record."""
    OUTPUT.mkdir(exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg", PYTHONPATH=str(ROOT))
    run([sys.executable, "-m", "pip", "check"], "dependencies", environment)
    converter = os.environ.get("GEOUNED_PYTHON", str(ROOT / ".geouned/bin/python"))
    run([converter, "-c", "import json, freecad, FreeCAD; from importlib.metadata import version; "
         "print(json.dumps({'geouned': version('geouned'), 'freecad': '.'.join(FreeCAD.Version()[:3])}))"],
        "geometry_dependencies", environment)
    run([sys.executable, "-m", "get_nuclear_data.prepare_chain"], "depletion_chain", environment)
    run([sys.executable, "-m", "pytest", "tests", "-q"], "core_tests", environment)
    with tempfile.TemporaryDirectory(prefix="diffusion-check-") as temporary:
        target = Path(temporary) / "diffusion_surrogate"
        for directory in ("checks", "fixtures"):
            shutil.copytree(DIFFUSION / directory, target / directory)
        (target / "results").mkdir()
        (target / ".cache").mkdir()
        for name in CHECKS:
            run([sys.executable, str(target / "checks" / name)], name.removesuffix(".py"), environment)
        for path in (target / "results").glob("*.json"):
            shutil.copy2(path, OUTPUT / path.name)
    run([sys.executable, "tests/check_native_openmc.py"], "native_openmc", environment)
    run([sys.executable, "tests/check_coupling.py"], "coupling", environment)
    run([sys.executable, "tests/check_wr1_design.py"], "wr1_design_mapping", environment)
    run([sys.executable, "tests/check_cad_geometry.py"], "cad_geometry", environment)
    import build123d, gmsh, numpy, openmc, scipy, torch
    record = {"passed": True, "python": sys.version.split()[0], "openmc": openmc.__version__,
              "numpy": numpy.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
              "build123d": build123d.__version__, "gmsh": gmsh.__version__,
              "geometry_converter": json.loads((OUTPUT / "geometry_dependencies.log").read_text().splitlines()[-1]),
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "core_tests": (OUTPUT / "core_tests.log").read_text().strip().splitlines()[-1],
              "native_check": "native_openmc.json", "depletion_check": "coupling_primitives.json",
              "cad_check": "cad_geometry/verification.json",
              "diffusion_checks": list(CHECKS), "scope": "Software and bounded numerical verification; no production search was resumed."}
    (OUTPUT / "installation.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
