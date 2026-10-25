"""Prepare all WR-1 CAD variants without running transport or optimization."""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import openmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coupling.execution import prepare_case
from designs.wr_1.model.case import study
from designs.wr_1.build_geometry import parameterized_geometry
from geometry import CadGeometry
from geometry.diffusion import discretize

OUTPUT = ROOT / "validation-output/cad_geometry"
SCENARIOS = range(1, 7)
OWNERSHIP_SAMPLES = 20


def check_case(step):
    """Compare a prepared scenario's actual CAD solids and both solver mappings.

    Parameters
    ----------
    step : int
        WR-1 scenario number, selecting its declared initial TOML inputs.

    Returns
    -------
    dict
        Volume errors in cm³, sampled ownership checks and the generated STEP path.
    """
    started = time.perf_counter()
    case = replace(study(step), work=OUTPUT / f"step_{step}")
    directory = prepare_case(case)
    data = json.loads((directory / "case.json").read_text())
    path = directory / "geometry.step"
    cad = CadGeometry.read_step(path)
    labels = [volume.label for volume in cad.volumes]
    assert cad.is_direct
    assert labels == [region["label"] for region in data["regions"]]
    actual = np.array([volume.shape.volume / 1000 for volume in cad.volumes])
    expected = np.array([region["volume_cm3"] for region in data["regions"]])
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=2e-8)
    live = parameterized_geometry(case.design(), data)
    mesh = discretize(live, 6, axial_subdivisions=1)
    represented = (mesh.fractions().T @ mesh.grid.volumes).detach().numpy()
    np.testing.assert_allclose(represented, expected, rtol=1e-10, atol=1e-8)

    openmc.reset_auto_ids()
    native = openmc.Geometry.from_xml(directory / "geometry.xml",
                                     materials=openmc.Materials.from_xml(directory / "materials.xml"))
    cells = tuple(native.get_all_cells().values())
    assert [cell.name for cell in cells] == labels
    rng = np.random.default_rng(4200 + step)
    vertices = live.domain.parameters["vertices"].detach().numpy()
    z = live.domain.parameters["z"].detach().numpy()
    for _ in range(OWNERSHIP_SAMPLES):
        r, s = rng.random(2)
        r = np.sqrt(r)
        xy = (1 - r) * vertices[0] + r * (1 - s) * vertices[1] + r * s * vertices[2]
        point = (*xy, float(rng.uniform(*z)))
        native_owners = [cell.name for cell in cells if point in cell.region]
        cad_owners = [volume.label for volume in cad.volumes
                      if any(solid.is_inside(tuple(10 * value for value in point)) for solid in volume.shape.solids())]
        assert len(native_owners) == 1 and native_owners == cad_owners, (step, point, native_owners, cad_owners)
    return {"step": step, "regions": len(labels), "passed": True,
            "max_cad_volume_error_cm3": float(np.max(abs(actual - expected))),
            "max_diffusion_volume_error_cm3": float(np.max(abs(represented - expected))),
            "ownership_points": OWNERSHIP_SAMPLES, "wall_s": time.perf_counter() - started,
            "step_file": str(path)}


def main():
    """Run bounded CAD checks and retain a readable verification record."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for step in SCENARIOS:
        record = check_case(step)
        records.append(record)
        print(f"Step {step}: {record['regions']} material volumes agree", flush=True)
    result = {"passed": True, "completed_utc": datetime.now(timezone.utc).isoformat(), "scenarios": records,
              "scope": "TOML preparation, actual STEP solid identities/volumes, native cell ownership and "
                       "diffusion volume closure. No transport history or reactor optimization was run."}
    (OUTPUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
