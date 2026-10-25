"""Discretize shared CAD using tetrahedra or exact analytical material cuts.

Actual build123d solids use a conformal Gmsh tetrahedral mesh. The optional live
box/axial construction retains exact moving-interface derivatives on Cartesian
and triangular-prism meshes. Both paths preserve the ordered material identities.
"""

from collections import defaultdict
from itertools import combinations
import math

import numpy as np
import torch

from .intersections import circle_triangle_area
from .model import CadGeometry, Shape


def _clip_polygon(vertices, boundary):
    """Clip a live polygon against a counterclockwise convex boundary.

    Parameters
    ----------
    vertices, boundary : torch.Tensor
        Ordered x/y coordinates in cm.

    Returns
    -------
    torch.Tensor
        Intersection polygon, retaining derivatives within its current topology.
    """
    result = list(vertices)
    for first, second in zip(boundary, torch.roll(boundary, -1, 0)):
        if not result:
            break
        edge = second - first
        points, result = result, []
        previous = points[-1]
        old = edge[0] * (previous[1] - first[1]) - edge[1] * (previous[0] - first[0])
        for current in points:
            distance = edge[0] * (current[1] - first[1]) - edge[1] * (current[0] - first[0])
            if bool(old.detach() >= 0) != bool(distance.detach() >= 0):
                result.append(previous + old / (old - distance) * (current - previous))
            if bool(distance.detach() >= 0):
                result.append(current)
            previous, old = current, distance
    return torch.stack(result) if result else vertices[:0]


def _rectangle(lower, upper):
    """Return live counterclockwise rectangle vertices.

    Parameters
    ----------
    lower, upper : torch.Tensor
        Opposite x/y corners in cm.

    Returns
    -------
    torch.Tensor
        Four vertices in cm.
    """
    return torch.stack((lower, torch.stack((upper[0], lower[1])), upper,
                        torch.stack((lower[0], upper[1]))))


def _polygon_area(vertices):
    """Compute oriented areas of counterclockwise polygons.

    Parameters
    ----------
    vertices : torch.Tensor
        Polygon coordinates in cm, with trailing axes (vertices, 2).

    Returns
    -------
    torch.Tensor
        Areas in cm² over the leading axes.
    """
    following = torch.roll(vertices, -1, -2)
    return (vertices[..., 0] * following[..., 1] - vertices[..., 1] * following[..., 0]).sum(-1) / 2


class _Intersections:
    """Integrate Boolean CAD constructions over an extruded background mesh."""

    def __init__(self, cad, grid):
        """Bind one live geometry evaluation to the solver's cell ordering.

        Parameters
        ----------
        cad : CadGeometry
            Physical geometry for this evaluation.
        grid : Grid3D or C6Grid
            Cartesian or triangular-prism mesh in cm, with z varying fastest.
        """
        self.cad = cad
        if hasattr(grid, "vertices"):
            self.polygons, self.z = grid.vertices, grid.z_edges
        else:
            x, y, self.z = grid.edges
            lower = torch.stack(torch.meshgrid(x[:-1], y[:-1], indexing="ij"), -1).reshape(-1, 2)
            upper = torch.stack(torch.meshgrid(x[1:], y[1:], indexing="ij"), -1).reshape(-1, 2)
            self.polygons = torch.stack([_rectangle(lo, hi) for lo, hi in zip(lower, upper)])
        self.areas = _polygon_area(self.polygons)
        self.terms, self.bounds, self.area_cache, self.intersections = {}, {}, {}, {}

    def nonempty(self, primitives):
        """Discard empty intersections before expanding a Boolean union.

        Parameters
        ----------
        primitives : tuple of Shape
            Convex primitive constraints, including internal axial slabs.

        Returns
        -------
        bool
            False only for a provably empty intersection. Decisions select the
            local intersection topology; retained dimensions remain live.
        """
        if primitives in self.intersections:
            return self.intersections[primitives]
        for shape in primitives:
            if shape not in self.bounds:
                if shape.kind == "slab":
                    z = shape.parameters["z"].detach().cpu().numpy()
                    self.bounds[shape] = np.array([[-np.inf, -np.inf, z[0]], [np.inf, np.inf, z[1]]])
                else:
                    self.bounds[shape] = np.array([value.detach().cpu().numpy() for value in shape.bounds()])
        bounds = np.stack([self.bounds[shape] for shape in primitives])
        valid = bool(np.all(bounds[:, 0].max(0) < bounds[:, 1].min(0)))
        circles = [shape for shape in primitives if shape.kind == "cylinder"]
        for left, right in combinations(circles, 2):
            p, q = left.parameters, right.parameters
            if float(torch.linalg.vector_norm(p["xy"] - q["xy"]).detach()) >= float((p["radius"] + q["radius"]).detach()):
                valid = False
        self.intersections[primitives] = valid
        return valid

    def multiply(self, first, second):
        """Intersect signed primitive expansions without exponential empty terms.

        Parameters
        ----------
        first, second : dict
            Primitive tuples mapped to signed integer multiplicities.

        Returns
        -------
        dict
            Combined expansion with repeated and empty terms removed.
        """
        result = defaultdict(int)
        for left, a in first.items():
            for right, b in second.items():
                key = tuple(sorted(set(left + right), key=id))
                if self.nonempty(key):
                    result[key] += a * b
        return {key: value for key, value in result.items() if value}

    def expand(self, shape):
        """Expand exact Boolean indicators into signed convex intersections.

        Parameters
        ----------
        shape : Shape
            Shared CAD construction.

        Returns
        -------
        dict
            Signed primitive intersections. Unions use inclusion-exclusion, so
            overlapping pieces are never silently double-counted.
        """
        if shape in self.terms:
            return self.terms[shape]
        if not shape.children:
            result = {(shape,): 1}
        elif shape.kind == "slice":
            slab = Shape("slab", shape.parameters)
            result = self.multiply(self.expand(shape.children[0]), {(slab,): 1})
        else:
            result = self.expand(shape.children[0])
            for child in shape.children[1:]:
                other = self.expand(child)
                overlap = self.multiply(result, other)
                if shape.kind == "intersection":
                    result = overlap
                else:
                    merged = defaultdict(int, result)
                    if shape.kind == "union":
                        for key, value in other.items():
                            merged[key] += value
                    elif shape.kind != "difference":
                        raise ValueError(f"Unsupported CAD operation: {shape.kind}")
                    for key, value in overlap.items():
                        merged[key] -= value
                    result = {key: value for key, value in merged.items() if value}
        self.terms[shape] = result
        return result

    def transverse_area(self, primitives):
        """Integrate one transverse intersection over all mesh polygons.

        Parameters
        ----------
        primitives : tuple of Shape
            Primitive constraints; the mesh already lies inside the CAD domain.

        Returns
        -------
        torch.Tensor
            Exact area in cm² per transverse cell.
        """
        key = tuple(shape for shape in primitives if shape is not self.cad.domain and shape.kind != "slab")
        if key in self.area_cache:
            return self.area_cache[key]
        circles = sorted((shape for shape in key if shape.kind == "cylinder"),
                         key=lambda shape: float(shape.parameters["radius"].detach()))
        circle = circles[0].parameters if circles else None
        for other in circles[1:]:
            p = other.parameters
            separation = torch.linalg.vector_norm(circle["xy"] - p["xy"])
            if float((separation + circle["radius"] - p["radius"]).detach()) > 1e-12:
                raise NotImplementedError("Diffusion intersections require disjoint or nested axial cylinders")
        constraints = []
        for shape in key:
            p = shape.parameters
            if shape.kind == "box":
                constraints.append(_rectangle(p["lower"][:2], p["upper"][:2]))
            elif shape.kind == "prism":
                constraints.append(p["vertices"])
        if not constraints:
            result = (self.areas if circle is None else
                      circle_triangle_area(self.polygons, circle["xy"], circle["radius"]))
        else:
            areas = []
            for polygon in self.polygons:
                clipped = polygon
                for boundary in constraints:
                    clipped = _clip_polygon(clipped, boundary)
                if len(clipped) < 3:
                    area = polygon.sum() * 0
                else:
                    area = (_polygon_area(clipped) if circle is None else
                            circle_triangle_area(clipped[None], circle["xy"], circle["radius"])[0])
                areas.append(area)
            result = torch.stack(areas)
        self.area_cache[key] = result
        return result

    def volume(self, shape):
        """Integrate a named material solid with exact axial overlap.

        Parameters
        ----------
        shape : Shape
            Material volume from the shared CAD construction.

        Returns
        -------
        torch.Tensor
            Represented volume in cm³, flattened with z varying fastest.
        """
        result = self.areas[:, None] * self.z.diff()[None, :] * 0
        for primitives, coefficient in self.expand(self.cad.domain & shape).items():
            bounds = [p.parameters["z"] if p.kind == "slab" else
                      torch.stack((p.bounds()[0][2], p.bounds()[1][2])) for p in primitives]
            bounds = torch.stack(bounds)
            low, high = bounds[:, 0].amax(), bounds[:, 1].amin()
            height = (torch.minimum(self.z[1:], high) - torch.maximum(self.z[:-1], low)).clamp_min(0)
            result = result + coefficient * self.transverse_area(primitives)[:, None] * height[None, :]
        return result.flatten()


def material_fractions(cad, grid):
    """Map the shared material volumes onto a diffusion mesh with live gradients.

    Parameters
    ----------
    cad : CadGeometry or str or pathlib.Path
        Parametric CAD model or a fixed STEP exported by this package.
    grid : Grid3D or C6Grid
        Mesh generated inside the same physical domain by discretize.

    Returns
    -------
    torch.Tensor
        Cell-by-material volume fractions in CAD volume order.

    Raises
    ------
    ArithmeticError
        Material solids overlap or leave gaps in the represented domain.
    NotImplementedError
        An intersection contains overlapping, non-nested cylinders.
    """
    cad = cad if isinstance(cad, CadGeometry) else CadGeometry.read_step(cad)
    integrator = _Intersections(cad, grid)
    result = torch.stack([integrator.volume(volume.shape) for volume in cad.volumes], -1) / grid.volumes[:, None]
    detached = result.detach()
    if float(detached.min()) < -2e-10 or float((detached.sum(-1) - 1).abs().max()) > 2e-10:
        raise ArithmeticError("CAD material volumes do not partition the diffusion domain")
    return result.clamp_min(0)


def _axial_edges(cad, subdivisions):
    """Subdivide the axial interfaces already present in the CAD construction.

    Parameters
    ----------
    cad : CadGeometry
        Shared physical geometry.
    subdivisions : int
        Number of mesh layers within each physical interval.

    Returns
    -------
    torch.Tensor
        Increasing live axial edges in cm.
    """
    pending = [cad.domain, *(volume.shape for volume in cad.volumes)]
    seen, heights = set(), []
    while pending:
        shape = pending.pop()
        if shape in seen:
            continue
        seen.add(shape)
        pending.extend(shape.children)
        if "z" in shape.parameters:
            heights.extend(shape.parameters["z"])
        elif shape.kind == "box":
            heights.extend((shape.parameters["lower"][2], shape.parameters["upper"][2]))
    edges = torch.stack(heights).sort().values
    edges = edges[torch.cat((torch.ones(1, dtype=torch.bool), edges.detach().diff() > 1e-12))]
    low, high = cad.domain.bounds()
    edges = edges[(edges.detach() >= low[2].detach()) & (edges.detach() <= high[2].detach())]
    weights = torch.arange(subdivisions, dtype=edges.dtype) / subdivisions
    return torch.cat(((edges[:-1, None] + edges.diff()[:, None] * weights).flatten(), edges[-1:]))


def discretize(cad, subdivisions=None, *, axial_subdivisions=1, max_size_cm=None,
               curvature_segments=32, coordinates=None, volume_rtol=1e-10):
    """Derive a solver mesh and material mapping from one CAD geometry factory.

    Parameters
    ----------
    cad : CadGeometry or CadMesh or callable or str or pathlib.Path
        Shared geometry, zero-argument live factory, or fixed STEP. A factory
        rebuilds the parameter graph for each fraction evaluation.
    subdivisions : tuple of int or int, optional
        Cartesian x/y/z cell counts for a box, or edge count for an equilateral
        positive-x 60-degree sector. Gmsh meshes the actual CAD sector face.
    axial_subdivisions : int, optional
        Layers per physical CAD interval for a sector mesh.
    max_size_cm : float, optional
        Required Gmsh edge-size target for direct build123d solids. A supplied
        CadMesh is reused unchanged, including its surface triangulation.
    curvature_segments : int, optional
        Curved-surface refinement for a newly generated CAD mesh.
    coordinates : callable or None, optional
        Live node coordinates in cm for a direct CAD mesh. The caller owns the
        explicit correspondence between these coordinates and its CAD design.
    volume_rtol : float, optional
        Allowed relative material-volume error during coupled preparation for
        tetrahedral meshes. Declare this accuracy requirement in the case's mesh
        inputs and refine the mesh to meet it. Analytical cuts retain exact closure.

    Returns
    -------
    DiffusionGeometry
        Numerical grid, live fraction callback and ordered material identities.

    Notes
    -----
    A callable updates internal material geometry on a fixed background mesh.
    Rebuild this mapping when changing the outer domain. Cartesian dimensions
    remain differentiable when discretizing inside each forward call. Sector meshes have a
    fixed outer domain; internal cylinders and axial slices retain derivatives.
    Direct build123d solids use conformal tetrahedra and FiniteElementDiffusion.
    Arbitrary CAD construction and remeshing do not supply autograd derivatives;
    use an explicit coordinate map when differentiating shape parameters.
    """
    from diffusion_surrogate.geometry import DiffusionGeometry
    from diffusion_surrogate.hexagonal import C6Grid
    from diffusion_surrogate.solver3d import Grid3D
    from diffusion_surrogate.finite_element import TetrahedralGrid
    from .meshing import CadMesh, tetrahedral_mesh

    initial = cad() if callable(cad) else cad
    if isinstance(initial, CadMesh):
        mesh, initial = initial, initial.cad
    else:
        mesh = None
    initial = initial if isinstance(initial, CadGeometry) else CadGeometry.read_step(initial)
    if initial.is_direct:
        if callable(cad):
            raise ValueError("Build the CAD solids first; supply an explicit coordinate map for shape gradients")
        if mesh is None:
            if max_size_cm is None:
                raise ValueError("Direct CAD meshing requires max_size_cm")
            mesh = tetrahedral_mesh(initial, max_size_cm=max_size_cm, curvature_segments=curvature_segments)
        grid = TetrahedralGrid(mesh, coordinates)
        assignment = torch.nn.functional.one_hot(torch.as_tensor(mesh.region_indices),
                                                 num_classes=len(initial.volumes)).to(torch.float64)
        return DiffusionGeometry(grid, lambda: assignment, tuple(v.label for v in initial.volumes),
                                 volume_rtol=volume_rtol)
    lower, upper = initial.domain.bounds()
    initial_bounds = torch.stack((lower, upper)).detach().clone()
    if initial.domain.kind == "box":
        names = ("x_min", "x_max", "y_min", "y_max", "bottom", "top")
        edges = tuple(lower[i] + (upper[i] - lower[i]) * torch.linspace(0., 1., n + 1, dtype=lower.dtype)
                      for i, n in enumerate(subdivisions))
        grid = Grid3D(edges, tuple(initial.boundaries[name]["type"] for name in names))
    elif initial.domain.kind == "prism":
        vertices = initial.domain.parameters["vertices"]
        a = vertices[1, 0]
        expected = vertices.new_tensor([[0., 0.], [1., -1 / math.sqrt(3)], [1., 1 / math.sqrt(3)]]) * a
        if vertices.shape != (3, 2) or not torch.allclose(vertices, expected):
            raise NotImplementedError("The sector diffusion mesh requires a positive-x equilateral CAD triangle")
        if vertices.requires_grad:
            raise NotImplementedError("A moving sector boundary requires a differentiable mesh adapter")
        z = _axial_edges(initial, axial_subdivisions)
        grid = C6Grid(float(a), subdivisions, z.detach() if callable(cad) else z,
                      symmetry_boundary=initial.boundaries["side/0"]["type"],
                      radial_boundary=initial.boundaries["side/1"]["type"],
                      axial_boundaries=tuple(initial.boundaries[name]["type"] for name in ("bottom", "top")),
                      cad_domain=initial.domain)
    else:
        raise NotImplementedError(f"No diffusion mesh adapter for CAD domain {initial.domain.kind!r}")

    def fractions():
        """Re-evaluate current material solids on the derived grid.

        Returns
        -------
        torch.Tensor
            Live cell-by-region fractions in CAD volume order.
        """
        current = cad() if callable(cad) else initial
        if callable(cad) and not torch.equal(torch.stack(current.domain.bounds()).detach(), initial_bounds):
            raise ValueError("Outer CAD domain changed; discretize again before constructing the solver")
        return material_fractions(current, grid)

    labels = tuple(volume.label for volume in initial.volumes)
    identity = ("cad", grid.shape, tuple(lower.detach().tolist()), tuple(upper.detach().tolist()),
                tuple(grid.boundaries), labels,
                tuple((grid.z_edges if hasattr(grid, "z_edges") else grid.edges[2]).detach().tolist()))
    return DiffusionGeometry(grid, fractions, labels, identity)
