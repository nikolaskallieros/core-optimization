"""Run GEOUNED in its separate FreeCAD environment using STEP/XML interchange."""

from importlib.metadata import version
from pathlib import Path


def convert():
    """Convert ``model.step`` in the working directory into ``geometry.xml``.

    Raises
    ------
    RuntimeError
        GEOUNED reports an uncertain decomposition or uses a release with a
        known conversion error. No geometry approximation is requested.

    Notes
    -----
    This module runs as a file in the converter environment. It deliberately
    does not import build123d, OpenMC or the main package's native libraries.
    """
    import geouned

    if version("geouned") in ("1.6.3", "1.6.4"):
        raise RuntimeError("GEOUNED 1.6.3/1.6.4 have known conversion errors; use environment-geouned.yml")
    formats = ("P_abc", "P_d", "P_xyz", "S_r", "S_xyz", "C_r", "C_xyz", "K_xyz", "K_tan2",
               "T_r", "T_xyz", "GQ_1to6", "GQ_7to9", "GQ_10")
    lengths = ("distance", "pln_distance", "cyl_distance", "sph_distance", "kne_distance", "tor_distance")
    angles = ("angle", "pln_angle", "cyl_angle", "kne_angle", "tor_angle")
    converter = geouned.CadToCsg(
        settings=geouned.Settings(voidGen=False, compSolids=True, simplify="no"),
        numeric_format=geouned.NumericFormat(**dict.fromkeys(formats, ".16e")),
        tolerances=geouned.Tolerances(**dict.fromkeys(lengths, 1e-7), **dict.fromkeys(angles, 1e-10),
                                     relativePrecision=1e-12, value=1e-10, min_area=0.))
    converter.load_step_file("model.step", spline_surfaces="stop")
    converter.start()
    if any(Path(converter.settings.outPath).glob("suspicious_solids/*")):
        raise RuntimeError("GEOUNED flagged an uncertain solid decomposition; conversion rejected")
    converter.export_csg(geometryName="geometry", outFormat=("openmc_xml",), cellSummaryFile=False)


if __name__ == "__main__":
    convert()
