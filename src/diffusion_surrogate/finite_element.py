"""Linear tetrahedral diffusion on general CAD meshes with Torch assembly.

The Galerkin discretization handles nonorthogonal tetrahedra. The converged
eigenproblem and its first-order implicit derivative reuse the sparse solver.
"""

import torch

from .solver3d import SparseDiffusion3D
from .solver import tensor


class TetrahedralGrid:
    """A conformal material mesh with optionally differentiable coordinates.

    Parameters
    ----------
    mesh : geometry.meshing.CadMesh
        Material and boundary connectivity derived from actual CAD solids.
    coordinates : callable or None, optional
        Zero-argument function returning live node coordinates in cm, in the
        mesh's node order. None retains the fixed CAD mesh. A shape update must
        preserve connectivity and CAD boundary conformity; remesh otherwise.
    """

    def __init__(self, mesh, coordinates=None):
        """Bind CAD connectivity and a physical coordinate map.

        Parameters
        ----------
        mesh : geometry.meshing.CadMesh
            Ordered mesh and region identities.
        coordinates : callable or None, optional
            Live node coordinates in cm.
        """
        self.mesh = mesh
        self._coordinates = coordinates
        self._points = tensor(mesh.points_cm)
        self.tetrahedra = torch.as_tensor(mesh.tetrahedra, dtype=torch.long)
        self.size = len(self.tetrahedra)
        self.shape = (self.size,)
        reference = self._points[self.tetrahedra]
        self._orientation = torch.linalg.det(reference[:, 1:] - reference[:, :1]).sign()
        self.boundary_names = tuple(dict.fromkeys(s["name"] for s in mesh.surfaces.values()
                                                  if s["type"] != "transmission"))

    @property
    def points(self):
        """Evaluate the live coordinate map.

        Returns
        -------
        torch.Tensor
            Node coordinates in cm, shaped (nodes, 3).
        """
        return self._points if self._coordinates is None else self._coordinates()

    @property
    def volumes(self):
        """Evaluate tetrahedron volumes without detaching coordinate derivatives.

        Returns
        -------
        torch.Tensor
            Element volumes in cm³, in CAD region-map order.

        Raises
        ------
        ValueError
            The supplied deformation inverts or collapses an element.
        """
        points = self.points[self.tetrahedra]
        determinants = torch.linalg.det(points[:, 1:] - points[:, :1])
        if bool((determinants.detach() * self._orientation <= 0).any()):
            raise ValueError("CAD mesh deformation inverted an element; remesh the updated solids")
        return determinants.abs() / 6


class FiniteElementDiffusion(SparseDiffusion3D):
    """Multigroup P1 diffusion on arbitrary conformal tetrahedral meshes.

    Cross sections are constant within each material element. The consistent
    mass matrix integrates reactions; the stiffness matrix integrates diffusion.
    Material coefficients and an explicit Torch coordinate map retain gradients.
    """

    def integration_weights(self):
        """Integrate nodal basis functions over the physical model.

        Returns
        -------
        torch.Tensor
            Nodal volumes in cm³.
        """
        return self.grid.points.new_zeros(len(self.grid.points)).index_add(
            0, self.grid.tetrahedra.flatten(), self.grid.volumes.repeat_interleave(4) / 4)

    def cell_flux(self, flux):
        """Integrate linear nodal flux over each material tetrahedron.

        Parameters
        ----------
        flux : torch.Tensor
            Nodal flux by energy group.

        Returns
        -------
        torch.Tensor
            Exact cell means of the piecewise-linear flux.
        """
        return flux[self.grid.tetrahedra].mean(1)

    def operators(self, fractions, macro, extrapolation=1.):
        """Assemble Galerkin loss, production and vacuum leakage operators.

        Parameters
        ----------
        fractions : torch.Tensor
            Element-by-region material map, normally the one-hot CAD assignment.
        macro : dict of str to torch.Tensor
            Regional macroscopic cross sections using the shared solver contract.
        extrapolation : float or torch.Tensor, optional
            Vacuum extrapolation factor, scalar or boundary-by-energy-group.
            Boundary order is ``grid.boundary_names``; reflective faces contribute
            no surface term.

        Returns
        -------
        tuple
            Sparse loss and production matrices, followed by nodal vacuum
            leakage weights in node-by-energy-group order.

        Notes
        -----
        The vacuum term is the weak-form Marshak condition J = phi/(2*factor).
        There is no orthogonality assumption or two-point face approximation.
        """
        points, cells = self.grid.points, self.grid.tetrahedra
        volumes, n = self.grid.volumes, len(points)
        vertices = points[cells]
        jacobian = vertices[:, 1:] - vertices[:, :1]
        reference = points.new_tensor([[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        gradients = reference @ torch.linalg.inv(jacobian).transpose(-1, -2)
        stiffness = volumes[:, None, None] * (gradients @ gradients.transpose(-1, -2))
        mass = volumes[:, None, None] * (torch.eye(4, dtype=points.dtype) + 1) / 20
        total, scatter, fission, diffusion = self._material_coefficients(fractions, macro)
        groups = total.shape[1]
        extrapolation = tensor(extrapolation).expand(len(self.grid.boundary_names), groups)
        row = cells[:, :, None].expand(-1, -1, 4).flatten()
        column = cells[:, None, :].expand(-1, 4, -1).flatten()
        ar, ac, av, br, bc, bv = [], [], [], [], [], []
        escape = points.new_zeros((n, groups))
        for outgoing in range(groups):
            for incoming in range(groups):
                values = -scatter[:, incoming, outgoing, None, None] * mass
                if outgoing == incoming:
                    values = values + total[:, outgoing, None, None] * mass + diffusion[:, outgoing, None, None] * stiffness
                ar.append(row + outgoing * n)
                ac.append(column + incoming * n)
                av.append(values.flatten())
                br.append(row + outgoing * n)
                bc.append(column + incoming * n)
                bv.append((fission[:, outgoing, incoming, None, None] * mass).flatten())
        for surface in self.grid.mesh.surfaces.values():
            if surface["type"] != "vacuum":
                continue
            faces = torch.as_tensor(surface["triangles"], dtype=torch.long)
            vertices = points[faces]
            areas = torch.linalg.vector_norm(torch.linalg.cross(vertices[:, 1] - vertices[:, 0],
                                                               vertices[:, 2] - vertices[:, 0]), dim=-1) / 2
            boundary = self.grid.boundary_names.index(surface["name"])
            coefficient = 1 / (2 * extrapolation[boundary])
            boundary_mass = areas[:, None, None] * (torch.eye(3, dtype=points.dtype) + 1) / 12
            r = faces[:, :, None].expand(-1, -1, 3).flatten()
            c = faces[:, None, :].expand(-1, 3, -1).flatten()
            for group in range(groups):
                ar.append(r + group * n)
                ac.append(c + group * n)
                av.append((coefficient[group] * boundary_mass).flatten())
            escape = escape.index_add(0, faces.flatten(),
                                      (areas[:, None, None] * coefficient / 3).expand(-1, 3, -1).reshape(-1, groups))

        return (self._assemble(ar, ac, av, n * groups),
                self._assemble(br, bc, bv, n * groups), escape)
