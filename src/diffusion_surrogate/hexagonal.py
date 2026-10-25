"""Sixfold triangular-prism meshes with conservative periodic connections."""
import math

import numpy as np
import torch

from .solver import tensor


class C6Grid:
    """Equilateral triangles extruded axially, with conservative C6 connections."""
    def __init__(self, apothem_cm, subdivisions, z_edges_cm, *, symmetry_boundary="periodic",
                 radial_boundary="vacuum", axial_boundaries=("vacuum", "vacuum"), cad_domain=None):
        """Build a triangular-prism mesh for one hexagonal symmetry sector.

        Parameters
        ----------
        apothem_cm : float
            Center-to-flat distance of the enclosing hexagon in cm.
        subdivisions : int
            Number of subdivisions along each triangular sector edge; at least two.
        z_edges_cm : array_like
            Increasing axial mesh edges in cm.
        symmetry_boundary : {'periodic', 'reflective'}, optional
            Boundary treatment on the two radial symmetry planes.
        radial_boundary : {'vacuum', 'reflective'}, optional
            Boundary treatment on the outer radial face.
        axial_boundaries : tuple of str, optional
            Bottom and top boundary types, each vacuum or reflective.
        cad_domain : geometry.Shape or None, optional
            Shared CAD sector. Gmsh meshes its actual lower face. None constructs
            the same sector for standalone numerical tests.

        Raises
        ------
        ValueError
            Geometry, subdivision count or symmetry boundary settings are invalid.
        ArithmeticError
            The triangular connectivity is nonmanifold.

        Notes
        -----
        Cells flatten in triangle/axial order, with the axial index varying fastest. Physical
        radial, bottom and top faces use the declared physical boundary conditions.
        """
        if symmetry_boundary not in ("periodic", "reflective"):
            raise ValueError("Invalid sector symmetry boundary")
        if len(axial_boundaries) != 2 or any(boundary not in ("vacuum", "reflective")
                                            for boundary in (radial_boundary, *axial_boundaries)):
            raise ValueError("Physical sector boundaries must be vacuum or reflective")
        if not isinstance(subdivisions, int) or subdivisions < 2:
            raise ValueError("At least two triangular subdivisions are required")
        n, a = subdivisions, float(apothem_cm)
        if not math.isfinite(a) or a <= 0:
            raise ValueError("Invalid moderator apothem")
        self.z_edges = tensor(z_edges_cm)
        if self.z_edges.ndim != 1 or bool((self.z_edges.diff() <= 0).any()):
            raise ValueError("Axial edges must increase")
        from geometry import prism
        from geometry.meshing import bottom_face, triangular_face
        if cad_domain is None:
            cad_domain = prism([[0., 0.], [a, -a / math.sqrt(3)], [a, a / math.sqrt(3)]],
                               [float(self.z_edges[0]), float(self.z_edges[-1])])
        points, triangles = triangular_face(bottom_face(cad_domain), n)
        lattice = np.rint(np.column_stack((points[:, 0] - math.sqrt(3) * points[:, 1],
                                          points[:, 0] + math.sqrt(3) * points[:, 1])) * n / (2 * a)).astype(int)
        indices = {tuple(position): i for i, position in enumerate(lattice)}
        self.vertices = tensor(points[triangles])
        self.triangle_ids = torch.tensor(triangles)
        self.points = tensor(points)
        centers = points[triangles].mean(1)
        side = 2 * a / (math.sqrt(3) * n)
        area, half = math.sqrt(3) * side ** 2 / 4, side / (2 * math.sqrt(3))
        self.triangle_area = area
        nz = len(self.z_edges) - 1
        self.shape, self.size = (len(triangles), nz), len(triangles) * nz
        self.volumes = (torch.full((len(triangles), 1), area, dtype=torch.float64)
                        * self.z_edges.diff()[None, :]).flatten()
        zc = (self.z_edges[:-1] + self.z_edges[1:]) / 2
        self.centers_cm = torch.cat((tensor(centers).repeat_interleave(nz, 0), zc.repeat(len(triangles))[:, None]), dim=1)
        edge_cells = {}
        for cell, triangle in enumerate(triangles):
            for j in range(3):
                edge = tuple(sorted((triangle[j], triangle[(j + 1) % 3])))
                edge_cells.setdefault(edge, []).append(cell)
        internal, periodic, outer = [], [], []
        for edge, cells in edge_cells.items():
            if len(cells) == 2:
                internal.append(cells)
            elif len(cells) != 1:
                raise ArithmeticError("Nonmanifold triangular mesh")
        for k in range(n):
            lower = tuple(sorted((indices[k, 0], indices[k + 1, 0])))
            upper = tuple(sorted((indices[0, k], indices[0, k + 1])))
            pair = (edge_cells[lower][0], edge_cells[upper][0])
            if pair[0] != pair[1] and symmetry_boundary == "periodic":
                periodic.append(pair)
            radial = tuple(sorted((indices[k, n - k], indices[k + 1, n - k - 1])))
            outer.append(edge_cells[radial][0])
        self.periodic_pairs = periodic
        dz = self.z_edges.diff()
        self.connections = []
        for pairs in (internal, periodic):
            pair = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
            first = (pair[:, 0, None] * nz + torch.arange(nz)).flatten()
            second = (pair[:, 1, None] * nz + torch.arange(nz)).flatten()
            self.connections.append((first, second, (side * dz).repeat(len(pair)),
                                     torch.full_like(first, half, dtype=torch.float64),
                                     torch.full_like(second, half, dtype=torch.float64)))
        ids = torch.arange(self.size).reshape(self.shape)
        first, second = ids[:, :-1].flatten(), ids[:, 1:].flatten()
        self.connections.append((first, second, torch.full_like(first, area, dtype=torch.float64),
                                 (dz[:-1] / 2).repeat(len(triangles)), (dz[1:] / 2).repeat(len(triangles))))
        face = (torch.tensor(outer)[:, None] * nz + torch.arange(nz)).flatten()
        empty = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float64),
                 torch.empty(0, dtype=torch.float64))
        # Six slots match the Cartesian solver's boundary parameters. Only radial,
        # bottom and top are physical boundary faces of this sector.
        self.faces = [(face, (side * dz).repeat(n), torch.full_like(face, half, dtype=torch.float64)),
                      empty, empty, empty,
                      (ids[:, 0], torch.full((len(triangles),), area, dtype=torch.float64), dz[0].expand(len(triangles)) / 2),
                      (ids[:, -1], torch.full((len(triangles),), area, dtype=torch.float64), dz[-1].expand(len(triangles)) / 2)]
        self.boundaries = (radial_boundary, "reflective", "reflective", "reflective", *axial_boundaries)
