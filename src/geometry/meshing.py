"""Gmsh discretization of faces taken from the shared build123d geometry."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile

import build123d as bd
import gmsh
import numpy as np


@contextmanager
def _gmsh_model():
    """Own a temporary Gmsh model without replacing a caller's model.

    Yields
    ------
    None
        A current empty model with terminal output disabled.
    """
    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize(readConfigFiles=False, interruptible=False)
    previous = gmsh.model.getCurrent() if gmsh.model.list() else None
    options = {name: gmsh.option.getNumber(name)
               for name in ("General.Terminal", "Mesh.TransfiniteTri", "Mesh.ElementOrder",
                            "Mesh.MeshSizeMin", "Mesh.MeshSizeMax", "Mesh.MeshSizeFromCurvature")}
    gmsh.model.add("core_geometry_mesh")
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    try:
        yield
    finally:
        gmsh.model.remove()
        if owned:
            gmsh.finalize()
        else:
            for name, value in options.items():
                gmsh.option.setNumber(name, value)
            if previous is not None:
                gmsh.model.setCurrent(previous)


def triangular_face(face, subdivisions):
    """Mesh a triangular CAD face using Gmsh's transfinite algorithm.

    Parameters
    ----------
    face : build123d.Face
        Planar triangular face in mm, parallel to the x/y plane.
    subdivisions : int
        Equal divisions along each CAD edge.

    Returns
    -------
    tuple of numpy.ndarray
        Transverse nodes in cm, shaped (n_nodes, 2), and counterclockwise triangle
        connectivity, shaped (n_triangles, 3).

    Notes
    -----
    The temporary BREP is only a Gmsh interchange buffer. The model's STEP file
    remains its single retained CAD artifact. Material interfaces are integrated
    on this background mesh by the diffusion adapter.
    """
    with tempfile.TemporaryDirectory(prefix="core-geometry-") as temporary, _gmsh_model():
        path = Path(temporary) / "face.brep"
        bd.export_brep(face, path)
        entities = gmsh.model.occ.importShapes(str(path), highestDimOnly=True)
        gmsh.model.occ.synchronize()
        surface, = [tag for dimension, tag in entities if dimension == 2]
        edges = gmsh.model.getBoundary([(2, surface)], oriented=False)
        if len(edges) != 3:
            raise ValueError("A triangular CAD face is required for the prism mesh")
        for _, edge in edges:
            gmsh.model.mesh.setTransfiniteCurve(edge, subdivisions + 1)
        gmsh.option.setNumber("Mesh.TransfiniteTri", 1)
        gmsh.model.mesh.setTransfiniteSurface(surface)
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(coordinates).reshape(-1, 3)[:, :2] / 10
        index = {int(tag): i for i, tag in enumerate(tags)}
        _, nodes = gmsh.model.mesh.getElementsByType(2)
        triangles = np.array([index[int(tag)] for tag in nodes], dtype=int).reshape(-1, 3)
    vertices = coordinates[triangles]
    first, second = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    reverse = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0] < 0
    triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
    return coordinates, triangles


def bottom_face(shape):
    """Select the lower transverse face of one extruded CAD domain.

    Parameters
    ----------
    shape : geometry.Shape
        Parametric extruded outer domain.

    Returns
    -------
    build123d.Face
        Actual CAD face in mm, preserving the model's outline.
    """
    faces = [face for face in shape.build().faces()
             if face.geom_type == bd.GeomType.PLANE and abs(face.normal_at().Z) > 1 - 1e-12]
    return min(faces, key=lambda face: face.center().Z)


@dataclass
class CadMesh:
    """Conformal tetrahedral and boundary connectivity from the shared CAD.

    Parameters
    ----------
    cad : CadGeometry
        Actual build123d solids with ordered material and boundary assignments.
    points_cm : numpy.ndarray
        Mesh vertices, shaped (nodes, 3), in cm.
    tetrahedra : numpy.ndarray
        Linear tetrahedron connectivity, shaped (elements, 4).
    region_indices : numpy.ndarray
        Material-region index for each tetrahedron.
    surfaces : dict
        Surface IDs mapped to triangle connectivity, adjacent solid senses and
        boundary name/type. Diffusion integrates its boundary terms on these faces.
    """

    cad: object
    points_cm: np.ndarray
    tetrahedra: np.ndarray
    region_indices: np.ndarray
    surfaces: dict


def tetrahedral_mesh(cad, *, max_size_cm, curvature_segments=32):
    """Mesh direct build123d solids with shared material interfaces using Gmsh.

    Parameters
    ----------
    cad : CadGeometry
        Nonoverlapping material solids in mm. Geometry is passed through BREP,
        avoiding binary coupling between Gmsh and build123d's CAD kernels.
    max_size_cm : float
        Maximum tetrahedron edge-size target in cm.
    curvature_segments : int, optional
        Target surface segments per full circle. Refinement controls the
        diffusion mesh's approximation of curved CAD surfaces.

    Returns
    -------
    CadMesh
        Conformal tetrahedra and their oriented material-interface triangles.

    Raises
    ------
    ValueError
        Material volumes overlap, a region is empty, or a boundary selection
        identifies an internal interface instead of an exterior surface.
    NotImplementedError
        A requested boundary condition is unavailable for the CAD mesh adapters.

    Notes
    -----
    CAD construction and meshing are not autograd operations. Shape derivatives
    require an explicit differentiable node-coordinate map in the diffusion
    adapter. Cross-section derivatives do not require such a map.
    """
    with tempfile.TemporaryDirectory(prefix="core-cad-") as temporary, _gmsh_model():
        inputs, owners, selections = [], [], []
        path = Path(temporary) / "shape.brep"
        for index, volume in enumerate(cad.volumes):
            if not volume.shape.solids():
                raise ValueError(f"Material region {volume.label!r} contains no solid")
            bd.export_brep(volume.shape, path)
            entities = gmsh.model.occ.importShapes(str(path), highestDimOnly=True)
            solids = [(dim, tag) for dim, tag in entities if dim == 3]
            inputs.extend(solids)
            owners.extend([index] * len(solids))
        for name, spec in cad.boundaries.items():
            if spec["type"] not in ("vacuum", "reflective"):
                raise NotImplementedError(f"CAD mesh boundary {name!r}: {spec['type']}")
            for face in spec["faces"]:
                bd.export_brep(face, path)
                for entity in gmsh.model.occ.importShapes(str(path), highestDimOnly=True):
                    if entity[0] == 2:
                        selections.append((entity, name, spec["type"]))
        tools = [entry[0] for entry in selections]
        if len(inputs) + len(tools) > 1:
            _, mapping = gmsh.model.occ.fragment(inputs, tools)
        else:
            mapping = [[entity] for entity in inputs]
        gmsh.model.occ.synchronize()
        solid_regions = {}
        for owner, fragments in zip(owners, mapping[:len(inputs)], strict=True):
            for dimension, tag in fragments:
                if dimension == 3:
                    if tag in solid_regions and solid_regions[tag] != owner:
                        raise ValueError("CAD material regions overlap; resolve the Boolean geometry first")
                    solid_regions[tag] = owner
        boundary_tags = {}
        for (_, name, kind), fragments in zip(selections, mapping[len(inputs):], strict=True):
            for dimension, tag in fragments:
                if dimension == 2:
                    if tag in boundary_tags and boundary_tags[tag] != (name, kind):
                        raise ValueError("A CAD face has conflicting boundary assignments")
                    boundary_tags[tag] = (name, kind)
        senses = {}
        for volume in solid_regions:
            for _, surface in gmsh.model.getBoundary([(3, volume)], oriented=True):
                sense = senses.setdefault(abs(surface), [0, 0])
                slot = 0 if surface > 0 else 1
                if sense[slot]:
                    raise ValueError("CAD material interfaces are not a conformal manifold")
                sense[slot] = volume
        for tag in boundary_tags:
            if all(senses[tag]):
                raise ValueError("A boundary condition was assigned to an internal material interface")
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size_cm * 10)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_segments)
        gmsh.model.mesh.generate(3)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        tags = np.asarray(tags)[order]
        points = np.asarray(coordinates).reshape(-1, 3)[order] / 10
        tetrahedra, regions = [], []
        for tag, region in solid_regions.items():
            _, nodes = gmsh.model.mesh.getElementsByType(4, tag)
            connectivity = np.searchsorted(tags, nodes).reshape(-1, 4)
            tetrahedra.append(connectivity)
            regions.extend([region] * len(connectivity))
        surfaces = {}
        for tag, sense in senses.items():
            _, nodes = gmsh.model.mesh.getElementsByType(2, tag)
            name, kind = boundary_tags.get(tag, (f"surface/{tag}", "transmission" if all(sense) else "vacuum"))
            surfaces[tag] = {"triangles": np.searchsorted(tags, nodes).reshape(-1, 3),
                             "sense": tuple(sense), "name": name, "type": kind}
    return CadMesh(cad, points, np.concatenate(tetrahedra), np.asarray(regions), surfaces)
