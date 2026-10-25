"""Three-dimensional multigroup diffusion and first-order implicit Torch AD.

The solver accepts arbitrary region fractions and macroscopic material tables.
Geometry builders and calibration parameters are separate from its equations.
Dense Torch and sparse SciPy solves share material mixing and physical responses;
both retain first derivatives through the converged eigenpair.
"""
from dataclasses import dataclass
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch

from .solver import circle_fractions, tensor


@dataclass
class Grid3D:
    """Nonuniform Cartesian cells defined by three edge arrays in cm.

    Faces are ordered x-low, x-high, y-low, y-high, z-low, z-high; flattened
    fields follow x/y/z array order, with z varying fastest."""
    edges: tuple
    boundaries: tuple = ('vacuum',)*6

    def __post_init__(self):
        """Build nonuniform Cartesian cell volumes and face connections.

        Raises
        ------
        ValueError
            Edge arrays or the six boundary conditions are invalid.
        """
        if len(self.edges) != 3 or len(self.boundaries) != 6:
            raise ValueError('Three edge arrays and six boundaries are required')
        if any(b not in ('vacuum', 'reflective') for b in self.boundaries):
            raise ValueError('Boundaries must be vacuum or reflective')
        self.edges = tuple(tensor(e) for e in self.edges)
        if any(e.ndim != 1 or len(e) < 2 or not bool(torch.isfinite(e).all()) or
               bool((e.diff() <= 0).any()) for e in self.edges):
            raise ValueError('Grid edges must be finite and strictly increasing')
        self.shape = tuple(len(e)-1 for e in self.edges)
        widths = tuple(e.diff() for e in self.edges)
        self.widths = widths
        self.volumes = (widths[0][:, None, None]*widths[1][None, :, None]*widths[2][None, None, :]).flatten()
        self.size = self.volumes.numel()
        ids = torch.arange(self.size).reshape(self.shape)
        self.connections, self.faces = [], []
        for axis in range(3):
            first, second = [slice(None)]*3, [slice(None)]*3
            first[axis], second[axis] = slice(None, -1), slice(1, None)
            a, b = ids[tuple(first)].flatten(), ids[tuple(second)].flatten()
            shape = [1, 1, 1]
            shape[axis] = self.shape[axis]
            width = widths[axis].reshape(shape).expand(self.shape).flatten()
            self.connections.append((a, b, self.volumes[a]/width[a], width[a]/2, width[b]/2))
            for side in (0, -1):
                selected = [slice(None)]*3
                selected[axis] = side
                face = ids[tuple(selected)].flatten()
                self.faces.append((face, self.volumes[face]/width[face], width[face]/2))

    def leakage(self, diffusion, extrapolation=1.):
        """Assemble conservative leakage for nonuniform Cartesian cells.

        Parameters
        ----------
        diffusion : torch.Tensor
            Diffusion coefficients in cm, shaped (cells,), (cells, 1) or (cells, 3) for
            independent axis components.
        extrapolation : float or torch.Tensor, optional
            Dimensionless vacuum factor, scalar or one value for each of six faces.

        Returns
        -------
        torch.Tensor
            Dense cell-by-cell leakage matrix for one energy group.
        """
        d = diffusion[:, None].expand(-1, 3) if diffusion.ndim == 1 else diffusion
        # A scalar isotropic cell value is conventionally supplied as (cells,1).
        if d.shape == (self.size, 1):
            d = d.expand(-1, 3)
        if d.shape != (self.size, 3):
            raise ValueError('Diffusion coefficients need one or three components per cell')
        result = d.new_zeros((self.size, self.size))
        diagonal = d.new_zeros(self.size)
        for axis, (a, b, area, half_a, half_b) in enumerate(self.connections):
            conductance = area/(half_a/d[a, axis]+half_b/d[b, axis])
            diagonal = diagonal.index_add(0, a, conductance).index_add(0, b, conductance)
            result = result.index_put((a, b), -conductance).index_put((b, a), -conductance)
        extrapolation = torch.as_tensor(extrapolation, dtype=d.dtype, device=d.device).expand(6)
        for index, ((face, area, half), boundary) in enumerate(zip(self.faces, self.boundaries)):
            if boundary == 'vacuum':
                value = d[face, index//2]
                diagonal = diagonal.index_add(0, face, area*value/(half+2*extrapolation[index]*value))
        return result+torch.diag(diagonal)


def cylinder_regions(grid, centers, radii, axial_bounds, background_edges=None):
    """Compute exact finite-cylinder and axial-background voxel fractions.

    Parameters
    ----------
    grid : Grid3D
        Fixed finite-volume grid with cell volumes, connections and boundary faces.
    centers : torch.Tensor
        Cylinder centers in cm with shape (n_cylinders, 2).
    radii : torch.Tensor
        Cylinder radii in cm with shape (n_cylinders,).
    axial_bounds : array_like
        Lower and upper z coordinates in cm, shaped (n_cylinders, 2).
    background_edges : array_like or None, optional
        Increasing z edges in cm partitioning the background; None uses one full-height
        background.

    Returns
    -------
    torch.Tensor
        Cell-by-region fractions, with cylinder segments followed by background intervals.

    Raises
    ------
    ValueError
        Cylinder geometry overlaps or fails the vessel and interval constraints.
    ArithmeticError
        The constructed fractions do not partition the vessel.

    Notes
    -----
    Each cylinder segment is an independently named region. Segments may share x/y centers
    provided their axial intervals do not overlap. Background regions are defined by
    optional z intervals. Region/material identities are external.
    """
    centers, radii, axial_bounds = tensor(centers), tensor(radii), tensor(axial_bounds)
    count = radii.numel()
    if centers.shape != (count, 2) or axial_bounds.shape != (count, 2):
        raise ValueError('Expected one center, radius and axial interval per segment')
    c, r, z = centers.detach(), radii.detach(), axial_bounds.detach()
    if not bool(torch.isfinite(c).all() & torch.isfinite(r).all() & torch.isfinite(z).all()):
        raise ValueError('Nonfinite geometry')
    if bool((r <= 0).any() | (z[:, 1] <= z[:, 0]).any()):
        raise ValueError('Cylinder radii and heights must be positive')
    for axis in range(2):
        if bool(((c[:, axis]-r < grid.edges[axis][0]) | (c[:, axis]+r > grid.edges[axis][-1])).any()):
            raise ValueError('Cylinder extends outside the vessel')
    if bool(((z[:, 0] < grid.edges[2][0]) | (z[:, 1] > grid.edges[2][-1])).any()):
        raise ValueError('Cylinder extends outside the axial vessel')
    separation = torch.cdist(c, c)+torch.eye(count, dtype=c.dtype)*1e30
    overlap = (torch.minimum(z[:, None, 1], z[None, :, 1]) >
               torch.maximum(z[:, None, 0], z[None, :, 0]))
    if bool(((separation < r[:, None]+r[None, :]) & overlap).any()):
        raise ValueError('Cylinder regions overlap')
    nx, ny, nz = grid.shape
    xy = circle_fractions(grid.edges[0], grid.edges[1], centers, radii).reshape(ny, nx, count).permute(1, 0, 2)
    ze = grid.edges[2]

    def length(low, high):
        """Compute fractional overlaps with the mesh's axial cells.

        Parameters
        ----------
        low : torch.Tensor
            Lower segment z coordinates in cm.
        high : torch.Tensor
            Upper segment z coordinates in cm.

        Returns
        -------
        torch.Tensor
            Axial-cell-by-segment length fractions.
        """
        return (torch.minimum(ze[1:, None], high)-torch.maximum(ze[:-1, None], low)).clamp_min(0)/ze.diff()[:, None]

    axial = length(axial_bounds[:, 0], axial_bounds[:, 1])
    pins = xy[:, :, None, :]*axial[None, None, :, :]
    background_edges = ze[[0, -1]] if background_edges is None else tensor(background_edges)
    if (not bool(torch.isfinite(background_edges).all()) or bool((background_edges.diff() <= 0).any()) or
            float(background_edges[0]) != float(ze[0]) or float(background_edges[-1]) != float(ze[-1])):
        raise ValueError('Background intervals must partition the full axial domain')
    backgrounds = []
    for low, high in zip(background_edges[:-1], background_edges[1:]):
        whole = length(low.reshape(1), high.reshape(1))[:, 0]
        inside = length(torch.maximum(axial_bounds[:, 0], low), torch.minimum(axial_bounds[:, 1], high))
        occupied = (xy[:, :, None, :]*inside[None, None, :, :]).sum(-1)
        backgrounds.append(whole[None, None, :]-occupied)
    fractions = torch.cat((pins, torch.stack(backgrounds, dim=-1)), dim=-1).reshape(grid.size, -1)
    if float(fractions.detach().min()) < -1e-11 or float((fractions.detach().sum(-1)-1).abs().max()) > 1e-11:
        raise ArithmeticError('Region volumes do not partition the vessel')
    return fractions.clamp_min(0.)


class Diffusion3D:
    """Solve regional multigroup diffusion with a converged-eigenpair adjoint.

    Region fractions and macroscopic tables are supplied by the caller.
    Implicit differentiation supports first derivatives of the converged solve."""
    def __init__(self, grid, *, tolerance=2e-11, max_iterations=30):
        """Configure the dense multigroup eigenproblem and convergence controls.

        Parameters
        ----------
        grid : Grid3D
            Fixed finite-volume grid with cell volumes, connections and boundary faces.
        tolerance : float, optional
            Convergence tolerance for the normalized eigenproblem residual.
        max_iterations : int, optional
            Maximum number of eigenproblem iterations before raising an error.
        """
        self.grid, self.tolerance, self.max_iterations = grid, tolerance, max_iterations
        self._guess = None

    @staticmethod
    def _material_coefficients(fractions, macro):
        """Mix regional cross sections into mesh-cell coefficients.

        Parameters
        ----------
        fractions : torch.Tensor
            Cell-by-region volume fractions retaining geometry derivatives.
        macro : dict of str to torch.Tensor
            Regional cross sections in cm⁻¹ and dimensionless fission spectra.
            Scattering axes are incoming then outgoing energy group.

        Returns
        -------
        tuple of torch.Tensor
            Total, scattering and fission-production cross sections in cm⁻¹,
            followed by diffusion coefficients in cm. Fission transfer axes are
            outgoing then incoming group; mixing preserves each region's spectrum.

        Raises
        ------
        ArithmeticError
            A mixed transport cross section is nonpositive.
        """
        total = fractions @ (macro['absorption'] + macro['scatter'].sum(-1))
        scatter = torch.einsum('cm,mgh->cgh', fractions, macro['nu_scatter'])
        fission = torch.einsum('cm,mo,mi->coi', fractions, macro['chi'], macro['nu_fission'])
        transport = fractions @ macro['transport']
        if bool((transport.detach() <= 0).any()):
            raise ArithmeticError('Nonpositive transport cross section')
        return total, scatter, fission, 1 / (3 * transport)

    def operators(self, fractions, macro, extrapolation=1.):
        """Assemble dense loss, fission and group leakage operators.

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
            Dense loss matrix, dense fission-production matrix and one leakage matrix per energy
            group.
        """
        total, scatter, fission, diffusion = self._material_coefficients(fractions, macro)
        groups = total.shape[1]
        extrapolation = tensor(extrapolation).expand(6, groups)
        leakage = [self.grid.leakage(diffusion[:, g:g+1], extrapolation[:, g]) for g in range(groups)]
        loss_rows, production_rows = [], []
        for outgoing in range(groups):
            loss_row, production_row = [], []
            for incoming in range(groups):
                block = torch.diag(-scatter[:, incoming, outgoing]*self.grid.volumes)
                if outgoing == incoming:
                    block = block+torch.diag(total[:, outgoing]*self.grid.volumes)+leakage[outgoing]
                loss_row.append(block)
                production_row.append(torch.diag(fission[:, outgoing, incoming]*self.grid.volumes))
            loss_rows.append(torch.cat(loss_row, dim=1))
            production_rows.append(torch.cat(production_row, dim=1))
        return torch.cat(loss_rows), torch.cat(production_rows), leakage

    @staticmethod
    def _jacobian(a, b, phi, lam, weight):
        """Assemble the bordered Jacobian of the normalized eigenproblem.

        Parameters
        ----------
        a : torch.Tensor
            Dense multigroup loss operator in group-major cell order.
        b : torch.Tensor
            Dense fission-production operator in the same order.
        phi : torch.Tensor
            Flattened group-major flux eigenvector.
        lam : torch.Tensor
            Reciprocal effective multiplication factor.
        weight : torch.Tensor
            Eigenvector normalization weights, repeated cell volumes by energy group.

        Returns
        -------
        torch.Tensor
            Jacobian for flux and reciprocal-keff updates, including the normalization row.
        """
        return torch.cat((torch.cat((a-lam*b, -(b@phi)[:, None]), dim=1),
                          torch.cat((weight, weight.new_zeros(1)))[None, :]))

    def _eigenpair(self, a, b, weight):
        """Find the positive principal eigenpair using a saved converged guess.

        Parameters
        ----------
        a : torch.Tensor
            Dense multigroup loss operator in group-major cell order.
        b : torch.Tensor
            Dense fission-production operator in the same order.
        weight : torch.Tensor
            Eigenvector normalization weights, repeated cell volumes by energy group.

        Returns
        -------
        tuple
            Detached flux eigenvector, reciprocal keff, iteration count and relative residual.

        Raises
        ------
        ArithmeticError
            No positive multiplying mode is found.
        RuntimeError
            The damped Newton iteration fails or does not converge.
        """
        a, b, weight = a.detach(), b.detach(), weight.detach()
        with torch.no_grad():
            if self._guess is not None and len(self._guess[0]) == len(a):
                phi, lam = (value.clone() for value in self._guess)
            else:
                phi = torch.ones_like(weight)/weight.sum()
                lu, pivots = torch.linalg.lu_factor(a)
                for _ in range(20):
                    updated = torch.linalg.lu_solve(lu, pivots, (b@phi)[:, None])[:, 0]
                    k = weight@updated
                    if not bool(torch.isfinite(k)) or float(k) <= 0:
                        raise ArithmeticError('No positive multiplying eigenmode')
                    phi, lam = updated/k, 1/k
            for iteration in range(self.max_iterations):
                r = torch.cat((a@phi-lam*(b@phi), (weight@phi-1).reshape(1)))
                scale = torch.linalg.vector_norm(a@phi)
                error = float(torch.linalg.vector_norm(r[:-1])/scale)
                if error < self.tolerance and abs(float(r[-1])) < self.tolerance:
                    break
                step = torch.linalg.solve(self._jacobian(a, b, phi, lam, weight), r)
                damping = 1.
                for _ in range(25):
                    trial_phi, trial_lam = phi-damping*step[:-1], lam-damping*step[-1]
                    trial_r = a@trial_phi-trial_lam*(b@trial_phi)
                    if (float(trial_phi.min()) >= -1e-15 and float(trial_lam) > 0 and
                            float(torch.linalg.vector_norm(trial_r)) < float(torch.linalg.vector_norm(r[:-1]))):
                        phi, lam = trial_phi, trial_lam
                        break
                    damping *= .5
                else:
                    raise RuntimeError('Positive eigenpair Newton step failed')
            else:
                raise RuntimeError(f'Three-dimensional eigenpair failed to converge: {error}')
            self._guess = (phi.clone(), lam.clone())
        return phi, lam, iteration+1, error

    def _validate_fractions(self, fractions):
        """Require a finite material partition of each mesh cell.

        Parameters
        ----------
        fractions : torch.Tensor
            Cell-by-region material volume fractions.

        Raises
        ------
        ValueError
            Dimensions, nonnegativity or unit row sums are invalid.
        """
        if fractions.ndim != 2 or fractions.shape[0] != self.grid.size:
            raise ValueError('Expected cell-by-region fractions')
        if (not bool(torch.isfinite(fractions.detach()).all()) or
                float(fractions.detach().min()) < -1e-12 or
                float((fractions.detach().sum(-1)-1).abs().max()) > 1e-10):
            raise ValueError('Invalid material-region partition')

    def _observables(self, fractions, macro, source_flux, leakage, power_W):
        """Normalize flux and assemble shared physical responses.

        Parameters
        ----------
        fractions : torch.Tensor
            Cell-by-region material volume fractions.
        macro : mapping of str to torch.Tensor
            Region/group data in cm⁻¹, with kappa_fission in eV/cm.
        source_flux : torch.Tensor
            Cell-by-group flux normalized per source neutron.
        leakage : torch.Tensor
            Outward leakage per source neutron in each energy group.
        power_W : float or torch.Tensor
            Prescribed total fission power in W.

        Returns
        -------
        dict of str to torch.Tensor
            Powers in W, power density in W/cm³, source rate in s⁻¹,
            flux/reaction responses per source, volumes in cm³ and balance error.

        Raises
        ------
        ArithmeticError
            Total heating is nonpositive or flux is materially negative.
        """
        integrated = source_flux*self.grid.volumes[:, None]
        region_flux = fractions.T@integrated
        heating = (macro['kappa_fission']*region_flux).sum(-1)
        total_heating = heating.sum()
        if float(total_heating.detach()) <= 0 or float(source_flux.detach().min()) < -1e-12:
            raise ArithmeticError('Nonphysical power or flux')
        power = tensor(power_W)*heating/total_heating
        cell_power = tensor(power_W)*(source_flux*(fractions@macro['kappa_fission'])).sum(-1)/total_heating
        extra = ((macro['nu_scatter']-macro['scatter'])*region_flux[:, :, None]).sum()
        balance = leakage.sum()+(macro['absorption']*region_flux).sum()-extra-1
        return {'power_W': power, 'power_density_W_cm3': cell_power.reshape(self.grid.shape),
                'source_rate_per_s': tensor(power_W)/(total_heating*1.602176634e-19),
                'flux_per_source': source_flux.reshape(*self.grid.shape, source_flux.shape[1]),
                'region_flux_per_source': region_flux,
                'capture_per_source': ((macro['absorption']-macro['fission'])*region_flux).sum(-1),
                'scatter_per_source': macro['scatter']*region_flux[:, :, None],
                'leakage_per_source': leakage, 'neutron_balance_error': balance,
                'volume_cm3': fractions.T@self.grid.volumes,
                'production_per_source': (macro['nu_fission']*region_flux).sum()}

    def solve(self, fractions, macro, *, power_W=100., extrapolation=1.):
        """Solve regional diffusion with first-order implicit differentiation.

        Parameters
        ----------
        fractions : torch.Tensor
            Cell-by-material-region volume fractions, nonnegative and summing to one in each
            cell.
        macro : dict of str to torch.Tensor
            Region-wise macroscopic reaction tables in cm⁻¹, kappa_fission in eV/cm, and
            dimensionless chi. Scattering axes are incoming then outgoing energy group.
        power_W : float or torch.Tensor, optional
            Total fission power in W used to normalize regional powers.
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
        The converged-eigenpair linearization supports first derivatives. It does not provide a
        consistent second derivative of the eigensolve.
        """
        started = time.perf_counter()
        self._validate_fractions(fractions)
        a, b, leakage = self.operators(fractions, macro, extrapolation)
        groups = len(a)//self.grid.size
        weight = self.grid.volumes.repeat(groups)
        phi0, lam0, iterations, error = self._eigenpair(a, b, weight)
        if torch.is_grad_enabled():
            # Implicit-function derivative at the converged root. The frozen
            # Jacobian is a numerical linearization, not fitted physics. This
            # supports first derivatives only; no second-derivative claim.
            with torch.no_grad():
                jacobian = self._jacobian(a.detach(), b.detach(), phi0, lam0, weight)
                lu, pivots = torch.linalg.lu_factor(jacobian)
            residual = torch.cat((a@phi0-lam0*(b@phi0), (weight@phi0-1).reshape(1)))
            update = torch.linalg.lu_solve(lu, pivots, residual[:, None])[:, 0]
            phi, lam = phi0-update[:-1], lam0-update[-1]
        else:
            phi, lam = phi0, lam0
        keff = 1/lam
        source_flux = (phi/(a@phi).sum()).reshape(groups, self.grid.size).T
        escape = torch.stack([operator.sum(0)@source_flux[:, g] for g, operator in enumerate(leakage)])
        fields = self._observables(fractions, macro, source_flux, escape, power_W)
        return {**fields, 'keff': keff,
                'eigen_residual': error, 'iterations': iterations,
                'solve_wall_s': time.perf_counter()-started}


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

    @staticmethod
    def _assemble(rows, columns, values, size):
        """Combine operator blocks into one coalesced sparse matrix.

        Parameters
        ----------
        rows, columns : list of torch.Tensor
            Paired block indices in group-major degree-of-freedom order.
        values : list of torch.Tensor
            Live coefficients paired with those indices.
        size : int
            Total number of degrees of freedom across all energy groups.

        Returns
        -------
        torch.Tensor
            Square sparse COO operator retaining coefficient derivatives.
        """
        return torch.sparse_coo_tensor(torch.stack((torch.cat(rows), torch.cat(columns))),
                                      torch.cat(values), (size, size)).coalesce()

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
        total, scatter, fission, diffusion = self._material_coefficients(fractions, macro)
        groups = total.shape[1]
        extrapolation = tensor(extrapolation).expand(6, groups)
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

        return (self._assemble(ar, ac, av, n * groups),
                self._assemble(br, bc, bv, n * groups), torch.stack(escapes, dim=-1))

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
