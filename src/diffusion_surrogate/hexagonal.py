"""Sixfold triangular-prism meshes and sparse multigroup diffusion."""
import math
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch

from geometry.intersections import circle_triangle_area

from .solver import tensor
from .solver3d import Diffusion3D


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


def _scipy(matrix):
    """Convert a detached Torch sparse matrix to SciPy CSC format.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse COO matrix on the CPU.

    Returns
    -------
    scipy.sparse.csc_matrix
        Coalesced matrix with explicit zero entries removed.
    """
    matrix = matrix.detach().coalesce()
    indices, values = matrix.indices().numpy(), matrix.values().numpy()
    result = sp.coo_matrix((values, indices), shape=matrix.shape).tocsc()
    result.eliminate_zeros()
    return result


def _matvec(matrix, vector):
    """Multiply sparse coefficients without a dense backward intermediate.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse COO matrix with differentiable stored coefficients.
    vector : torch.Tensor
        Dense right-hand vector.

    Returns
    -------
    torch.Tensor
        Dense matrix-vector product retaining derivatives through nonzero entries.
    """
    # Explicit nonzero contraction avoids an N-by-N intermediate in Torch's
    # sparse matrix-multiply backward. Only stored coefficients are differentiated.
    matrix = matrix.coalesce()
    row, column = matrix.indices()
    return vector.new_zeros(matrix.shape[0]).index_add(0, row, matrix.values() * vector[column])


class _FrozenSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, factor):
        """Solve a residual using a frozen sparse factorization.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context used to retain the factorization.
        residual : torch.Tensor
            CPU residual vector to solve against.
        factor : scipy.sparse.linalg.SuperLU
            Fixed factorization of the bordered eigenproblem Jacobian.

        Returns
        -------
        torch.Tensor
            CPU solution vector.
        """
        ctx.factor = factor
        return torch.from_numpy(factor.solve(residual.detach().numpy()))

    @staticmethod
    def backward(ctx, gradient):
        """Apply the transposed frozen solve to an output cotangent.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context holding the forward sparse factorization.
        gradient : torch.Tensor
            Incoming gradient with respect to the solve result.

        Returns
        -------
        tuple
            Residual gradient and None for the nondifferentiated factorization.
        """
        return torch.from_numpy(ctx.factor.solve(gradient.detach().numpy(), trans="T")), None


class SparseDiffusion3D(Diffusion3D):
    """Finite-volume diffusion with sparse LU and implicit differentiation."""
    def integration_weights(self):
        """Return weights for initializing the eigenvector.

        Returns
        -------
        torch.Tensor
            Physical integration weights in degree-of-freedom order.
        """
        return self.grid.volumes

    def cell_flux(self, flux):
        """Map solver degrees of freedom to material-cell mean fluxes.

        Parameters
        ----------
        flux : torch.Tensor
            Source-normalized flux by degree of freedom and energy group.

        Returns
        -------
        torch.Tensor
            Cell-mean fluxes; finite-volume unknowns already have this meaning.
        """
        return flux

    def operators(self, fractions, macro, extrapolation=1.):
        """Assemble sparse multigroup loss and fission operators.

        Parameters
        ----------
        fractions : torch.Tensor
            Cell-by-material-region volume fractions, nonnegative and summing to one in each
            cell.
        macro : dict of str to torch.Tensor
            Region-wise macroscopic reaction tables in cm⁻¹, kappa_fission in eV/cm, and
            dimensionless chi. Scattering axes are incoming then outgoing energy group.
        extrapolation : float or torch.Tensor, optional
            Dimensionless Marshak vacuum-extrapolation factor, broadcastable to six boundary
            slots by energy group.

        Returns
        -------
        tuple
            Torch sparse loss and production matrices, plus cell-by-group vacuum escape
            coefficients.

        Raises
        ------
        ArithmeticError
            A mixed transport cross section is nonpositive; no artificial void cross section is
            inserted.
        """
        n, volumes = self.grid.size, self.grid.volumes
        total = fractions @ (macro["absorption"] + macro["scatter"].sum(-1))
        scatter = torch.einsum("cm,mgh->cgh", fractions, macro["nu_scatter"])
        transport = fractions @ macro["transport"]
        if bool((transport.detach() <= 0).any()):
            raise ArithmeticError("Nonpositive mixed transport cross section; no void epsilon is inserted")
        diffusion, groups = 1 / (3 * transport), total.shape[1]
        extrapolation = tensor(extrapolation).expand(6, groups)
        fission = torch.einsum("cm,mo,mi->coi", fractions, macro["chi"], macro["nu_fission"])
        ar, ac, av, br, bc, bv, escapes = [], [], [], [], [], [], []
        ids = torch.arange(n)
        for outgoing in range(groups):
            diagonal, escape = total.new_zeros(n), total.new_zeros(n)
            for first, second, area, half_a, half_b in self.grid.connections:
                conductance = area / (half_a / diffusion[first, outgoing] + half_b / diffusion[second, outgoing])
                diagonal = diagonal.index_add(0, first, conductance).index_add(0, second, conductance)
                ar.extend((first + outgoing * n, second + outgoing * n))
                ac.extend((second + outgoing * n, first + outgoing * n))
                av.extend((-conductance, -conductance))
            for face_index, ((face, area, half), boundary) in enumerate(zip(self.grid.faces, self.grid.boundaries)):
                if boundary == "vacuum":
                    value = diffusion[face, outgoing]
                    conductance = area * value / (half + 2 * extrapolation[face_index, outgoing] * value)
                    escape = escape.index_add(0, face, conductance)
            diagonal = diagonal + escape
            escapes.append(escape)
            for incoming in range(groups):
                values = -scatter[:, incoming, outgoing] * volumes
                if outgoing == incoming:
                    values = values + total[:, outgoing] * volumes + diagonal
                ar.append(ids + outgoing * n)
                ac.append(ids + incoming * n)
                av.append(values)
                br.append(ids + outgoing * n)
                bc.append(ids + incoming * n)
                bv.append(fission[:, outgoing, incoming] * volumes)

        def sparse(rows, columns, values):
            """Combine operator blocks into one coalesced sparse matrix.

            Parameters
            ----------
            rows : list of torch.Tensor
                Row-index blocks in group-major order.
            columns : list of torch.Tensor
                Column-index blocks paired with rows.
            values : list of torch.Tensor
                Coefficient blocks paired with row and column indices.

            Returns
            -------
            torch.Tensor
                Coalesced sparse COO multigroup operator.
            """
            return torch.sparse_coo_tensor(torch.stack((torch.cat(rows), torch.cat(columns))),
                                           torch.cat(values), (n * groups, n * groups)).coalesce()
        return sparse(ar, ac, av), sparse(br, bc, bv), torch.stack(escapes, dim=-1)

    @staticmethod
    def _border(a, b, phi, lam, weight):
        """Build the sparse bordered eigenproblem Jacobian.

        Parameters
        ----------
        a : scipy.sparse.csc_matrix
            Multigroup loss operator in group-major cell order.
        b : scipy.sparse.csc_matrix
            Fission-production operator in the same order.
        phi : numpy.ndarray
            Flattened group-major flux eigenvector.
        lam : float
            Reciprocal effective multiplication factor.
        weight : numpy.ndarray
            Eigenvector normalization weights in group-major cell order.

        Returns
        -------
        scipy.sparse.csc_matrix
            Loss-minus-production Jacobian augmented by eigenvalue and normalization
            columns/rows.
        """
        return sp.bmat([[a - lam * b, sp.csc_matrix(-(b @ phi)[:, None])],
                        [sp.csc_matrix(weight[None, :]), sp.csc_matrix((1, 1))]], format="csc")

    def _eigenpair_sparse(self, a, b, weight):
        """Find the positive principal eigenpair with sparse damped Newton steps.

        Parameters
        ----------
        a : scipy.sparse.csc_matrix
            Multigroup loss operator in group-major cell order.
        b : scipy.sparse.csc_matrix
            Fission-production operator in the same order.
        weight : numpy.ndarray
            Eigenvector normalization weights in group-major cell order.

        Returns
        -------
        tuple
            Flux eigenvector, reciprocal keff, iteration count, relative residual and the
            selected one-entry normalization weights.

        Raises
        ------
        ArithmeticError
            No positive multiplying mode is found.
        RuntimeError
            A positive Newton step fails or convergence is not reached.
        """
        if self._guess is not None and len(self._guess[0]) == len(weight):
            phi, lam = self._guess[0].copy(), self._guess[1]
        else:
            phi, factor = np.ones_like(weight) / weight.sum(), splu(a)
            for _ in range(20):
                update = factor.solve(b @ phi)
                k = weight @ update
                if not np.isfinite(k) or k <= 0:
                    raise ArithmeticError("No positive multiplying eigenmode")
                phi, lam = update / k, 1 / k
            self._anchor = int(np.argmax(phi))
            phi = phi / phi[self._anchor]
        # An eigenvector's arbitrary amplitude is fixed at one positive entry.
        # Unlike a dense normalization row, this preserves sparse factorization.
        # Physical flux and power are normalized independently below.
        weight = np.zeros_like(weight)
        weight[self._anchor] = 1.
        for iteration in range(self.max_iterations):
            residual = np.append(a @ phi - lam * (b @ phi), weight @ phi - 1)
            error = np.linalg.norm(residual[:-1]) / np.linalg.norm(a @ phi)
            if error < self.tolerance and abs(residual[-1]) < self.tolerance:
                self._guess = phi.copy(), lam
                return phi, lam, iteration + 1, float(error), weight
            step = splu(self._border(a, b, phi, lam, weight)).solve(residual)
            damping = 1.
            for _ in range(25):
                trial_phi, trial_lam = phi - damping * step[:-1], lam - damping * step[-1]
                if (trial_phi.min() >= -1e-15 and trial_lam > 0
                        and np.linalg.norm(a @ trial_phi - trial_lam * (b @ trial_phi)) < np.linalg.norm(residual[:-1])):
                    phi, lam = trial_phi, trial_lam
                    break
                damping *= .5
            else:
                raise RuntimeError("Sparse positive-eigenpair Newton step failed")
        raise RuntimeError(f"Sparse diffusion eigenpair did not converge: {error}")

    def solve(self, fractions, macro, *, power_W=100., extrapolation=1.):
        """Solve sparse regional diffusion and attach its implicit first derivative.

        Parameters
        ----------
        fractions : torch.Tensor
            Cell-by-material-region volume fractions, nonnegative and summing to one in each
            cell.
        macro : dict of str to torch.Tensor
            Region-wise macroscopic reaction tables in cm⁻¹, kappa_fission in eV/cm, and
            dimensionless chi. Scattering axes are incoming then outgoing energy group.
        power_W : float or torch.Tensor, optional
            Total fission power in W used to normalize regional powers and source rate.
        extrapolation : float or torch.Tensor, optional
            Dimensionless Marshak vacuum-extrapolation factor, broadcastable to six boundary
            slots by energy group.

        Returns
        -------
        dict
            Effective multiplication factor, region powers in W, cell power density in W/cm³,
            source-normalized flux and reaction observables, region volumes in cm³, balance
            residuals, iteration count and timings.

        Notes
        -----
        Flux and reaction observables are normalized per source neutron. source_rate_per_s
        converts them to the requested physical power. The frozen-factor adjoint supports first
        derivatives only.
        """
        started = time.perf_counter()
        self._validate_fractions(fractions)
        a, b, escape = self.operators(fractions, macro, extrapolation)
        operators_done = time.perf_counter()
        degrees = escape.shape[0]
        groups = a.shape[0] // degrees
        weight = self.integration_weights().detach().repeat(groups)
        sa, sb = _scipy(a), _scipy(b)
        phi0, lam0, iterations, error, normalization = self._eigenpair_sparse(sa, sb, weight.numpy())
        weight = tensor(normalization)
        phi0, lam0 = tensor(phi0), tensor(lam0)
        eigen_done = time.perf_counter()
        if torch.is_grad_enabled():
            factor = splu(self._border(sa, sb, phi0.numpy(), float(lam0), weight.numpy()))
            residual = torch.cat((_matvec(a, phi0) - lam0 * _matvec(b, phi0), (weight @ phi0 - 1).reshape(1)))
            # Attach the implicit derivative without changing the converged
            # forward root when gradient tracking is enabled.
            update = _FrozenSolve.apply(residual - residual.detach(), factor)
            phi, lam = phi0 - update[:-1], lam0 - update[-1]
        else:
            phi, lam = phi0, lam0
        derivative_done = time.perf_counter()
        source_flux = (phi / _matvec(a, phi).sum()).reshape(groups, degrees).T
        leakage = (escape * source_flux).sum(0)
        fields = self._observables(fractions, macro, self.cell_flux(source_flux), leakage, power_W)
        return {**fields, "keff": 1 / lam, "eigen_residual": error,
                "iterations": iterations, "solve_wall_s": time.perf_counter() - started,
                "operators_wall_s": operators_done - started, "eigen_wall_s": eigen_done - operators_done,
                "implicit_factor_wall_s": derivative_done - eigen_done}
