"""Convert build123d material solids into OpenMC analytic geometry with GEOUNED."""

from dataclasses import dataclass
from functools import reduce
import operator
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import openmc
import build123d as bd
from lxml import etree
from OCP.BRepAdaptor import BRepAdaptor_Surface

from .model import CadGeometry


@dataclass
class OpenMCGeometry:
    """OpenMC objects derived from one material-volume CAD model.

    Parameters
    ----------
    geometry : openmc.Geometry
        Native geometry with cell order matching the CAD material-volume order.
    cells : tuple of openmc.Cell
        One native cell per named material region.
    boundaries : dict
        Domain-face names mapped to their native surfaces.
    source_bounds_cm : tuple
        Per-region tuples of finite lower/upper boxes for sampling each connected
        material piece in cm. The native cell rejects points outside the solid.
    """

    geometry: openmc.Geometry
    cells: tuple
    boundaries: dict
    source_bounds_cm: tuple


def _matching_surface(face, surfaces):
    """Match a CAD face to an analytic surface using points and normal offsets.

    Parameters
    ----------
    face : build123d.Face
        Actual CAD face in mm; samples lie on its underlying analytic surface.
    surfaces : mapping of int to openmc.Surface
        GEOUNED's surfaces in cm.

    Returns
    -------
    int
        Matching native surface identity.

    Raises
    ------
    ValueError
        The conversion omitted or changed a physical surface.
    """
    samples = [(np.array(tuple(face.position_at(u, v))) / 10,
                np.array(tuple(face.normal_at(u=u, v=v))))
               for u in (.17, .43, .79) for v in (.19, .47, .83)]
    for identity, surface in surfaces.items():
        for point, normal in samples:
            slope = abs(surface.evaluate(point + 1e-5 * normal) - surface.evaluate(point - 1e-5 * normal)) / 2e-5
            if slope == 0 or abs(surface.evaluate(point)) > 1e-8 * slope:
                break
        else:
            return identity
    raise ValueError(f"GEOUNED did not preserve a {face.geom_type} CAD surface")


def _boundaries(cad, surfaces, root):
    """Apply CAD boundary selections and inward periodic plane orientations.

    Parameters
    ----------
    cad : CadGeometry
        Actual material solids and named boundary faces.
    surfaces : dict of int to openmc.Surface
        Converted surfaces, initially all transmission boundaries.
    root : lxml.etree._Element
        Converted cell expressions, updated if a periodic plane is reversed.

    Returns
    -------
    dict
        Boundary names mapped to native surfaces. Unselected exterior faces are
        vacuum. A native surface cannot carry different conditions on its patches.
    """
    solids = [solid for volume in cad.volumes for solid in volume.shape.solids()]
    outside = solids[0].fuse(*solids[1:]) if len(solids) > 1 else solids[0]
    total_volume = sum(solid.volume for solid in solids)
    if abs(total_volume - outside.volume) > 1e-9 * total_volume:
        raise ValueError("CAD material solids overlap; OpenMC requires a material partition")
    groups = {}
    for solid in solids:
        for face in solid.faces():
            identity = _matching_surface(face, surfaces)
            point = np.array(tuple(face.position_at(.43, .47))) / 10
            normal = np.array(tuple(face.normal_at(u=.43, v=.47)))
            positive = surfaces[identity].evaluate(point + 1e-5 * normal) > 0
            groups.setdefault(identity, {True: [], False: []})[positive].append(face)
    exterior = {}
    for identity, oriented in groups.items():
        faces = oriented[True] + oriented[False]
        area, internal = sum(face.area for face in faces), 0.
        if all(oriented.values()):
            common = bd.Compound(oriented[True]) & bd.Compound(oriented[False])
            internal = 2 * common.area if common is not None else 0.
        if abs(area - internal) <= 1e-8 * area:
            continue
        # A native surface is global: a boundary tag must not affect a different
        # internal patch on the same plane or curved analytic surface.
        if internal > 1e-8 * area:
            raise NotImplementedError("An OpenMC surface spans exterior and internal CAD faces; split that boundary")
        exterior[identity] = faces
        surfaces[identity].boundary_type = "vacuum"

    boundaries, assigned, reversed_planes = {}, {}, set()
    for name, spec in cad.boundaries.items():
        selected = {}
        for face in spec["faces"]:
            identity = _matching_surface(face, surfaces)
            selected.setdefault(identity, []).append(face)
        if spec["type"] == "periodic" and len(selected) != 1:
            raise ValueError("Each periodic boundary must select one plane")
        for identity, faces in selected.items():
            outer_faces = exterior[identity]
            coverage = sum((common.area if (common := outer & face) is not None else 0.)
                           for outer in outer_faces for face in faces)
            area = sum(face.area for face in outer_faces)
            if coverage < area * (1 - 1e-8):
                raise NotImplementedError("A boundary selection covers only part of an OpenMC surface")
            if identity in assigned:
                raise ValueError(f"Boundary selections {assigned[identity]} and {name} share an OpenMC surface")
            assigned[identity] = name
            surface = surfaces[identity]
            surface.boundary_type = spec["type"]
            if spec["type"] == "periodic":
                if not isinstance(surface, openmc.Plane):
                    raise NotImplementedError("OpenMC periodic CAD boundaries must be planar")
                point = np.array(tuple(faces[0].position_at(.43, .47))) / 10
                normal = np.array(tuple(faces[0].normal_at(u=.43, v=.47)))
                if surface.evaluate(point + 1e-5 * normal) > 0:
                    reversed_planes.add(identity)
                # Use one general inward plane even when GEOUNED chose X/Y/ZPlane.
                surface = openmc.Plane(a=-normal[0], b=-normal[1], c=-normal[2],
                                       d=-normal @ point, boundary_type="periodic")
                surfaces[identity] = surface
            boundaries[name if len(selected) == 1 else f"{name}/{identity}"] = surface
    for identity in exterior.keys() - assigned.keys():
        boundaries[f"exterior/{identity}"] = surfaces[identity]
    for name, spec in cad.boundaries.items():
        if spec["type"] == "periodic":
            boundaries[name].periodic_surface = boundaries[spec["pair"]]
    if reversed_planes:
        import re
        for cell in root.findall("cell"):
            cell.set("region", re.sub(r"-?\d+", lambda m: str(-int(m[0]) if abs(int(m[0])) in reversed_planes
                                                            else int(m[0])), cell.get("region")))
    return boundaries


def from_cad(cad, materials, *, converter_python=None):
    """Convert the actual CAD solids through STEP and GEOUNED to OpenMC CSG.

    Parameters
    ----------
    cad : CadGeometry or str or pathlib.Path
        Shared build123d material solids, an analytical construction built into
        solids, or a STEP snapshot. The native model comes from the actual BREP.
    materials : mapping of str to openmc.Material or None
        Native fills keyed by region label or material key. A region label takes
        precedence, allowing independent depleted compositions. None means void.
    converter_python : str or pathlib.Path or None, optional
        Interpreter containing GEOUNED/FreeCAD. Defaults to ``GEOUNED_PYTHON`` or
        ``.geouned/bin/python`` under ``runtime.paths.PROJECT_ROOT``.

    Returns
    -------
    OpenMCGeometry
        Ordered native cells, physical boundaries and source sampling boxes.

    Raises
    ------
    NotImplementedError
        A CAD surface cannot be represented analytically by this converter.
        Unsupported geometry is rejected; it is never tessellated or simplified.

    Notes
    -----
    FreeCAD runs in a separate process to isolate its native libraries from
    build123d and OpenMC. Temporary STEP labels carry region identities through
    conversion; they do not define a second geometry. Conversion does not retain
    PyTorch derivatives. The live diffusion parameterization supplies those.
    """
    from runtime.paths import PROJECT_ROOT

    cad = (cad if isinstance(cad, CadGeometry) else CadGeometry.read_step(cad)).as_solids()
    supported = {bd.GeomType.PLANE, bd.GeomType.CYLINDER, bd.GeomType.CONE, bd.GeomType.SPHERE, bd.GeomType.TORUS}
    children, sources = [], []
    for index, volume in enumerate(cad.volumes, 1):
        boxes = []
        for solid in volume.shape.solids():
            for face in solid.faces():
                if face.geom_type not in supported:
                    raise NotImplementedError(f"GEOUNED cannot preserve {face.geom_type} in region {volume.label}")
                if face.geom_type == bd.GeomType.TORUS:
                    axis = BRepAdaptor_Surface(face.wrapped).Torus().Axis().Direction().Coord()
                    if max(abs(value) for value in axis) < 1 - 1e-12:
                        raise NotImplementedError("GEOUNED's OpenMC writer requires axis-aligned torus surfaces")
            bounds = solid.bounding_box()
            boxes.append((np.array(tuple(bounds.min)) / 10, np.array(tuple(bounds.max)) / 10))
        children.append(bd.Compound(list(volume.shape.solids()), label=f"region_{index}_m{index}_"))
        sources.append(tuple(boxes))
    python = Path(converter_python or os.environ.get("GEOUNED_PYTHON", PROJECT_ROOT / ".geouned/bin/python"))
    with tempfile.TemporaryDirectory(prefix="cad-csg-") as temporary:
        directory = Path(temporary)
        bd.export_step(bd.Compound(children=children), directory / "model.step")
        result = subprocess.run([str(python.expanduser().resolve()), str(Path(__file__).with_name("_geouned.py"))],
                                cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode or not (directory / "geometry.xml").exists():
            raise RuntimeError(f"GEOUNED conversion failed:\n{result.stdout}")
        root = etree.parse(str(directory / "geometry.xml")).getroot()
    surfaces = {}
    for element in root.findall("surface"):
        element.set("boundary", "transmission")
        surface = openmc.Surface.from_xml_element(element)
        surfaces[surface.id] = surface
    boundaries = _boundaries(cad, surfaces, root)
    regions = {}
    for element in root.findall("cell"):
        identity = int(element.get("material"))
        regions.setdefault(identity, []).append(openmc.Region.from_expression(element.get("region"), surfaces))
    if set(regions) != set(range(1, len(cad.volumes) + 1)):
        raise ValueError("GEOUNED changed the material-region identities")
    cells = tuple(openmc.Cell(index, name=volume.label,
                             fill=materials[volume.label if volume.label in materials else volume.material],
                             region=reduce(operator.or_, regions[index]))
                  for index, volume in enumerate(cad.volumes, 1))
    return OpenMCGeometry(openmc.Geometry(cells), cells, boundaries, tuple(sources))

