"""Multigroup finite-volume diffusion, with ordinary Torch reverse derivatives.

Pins are extruded cylinders with reflecting axial boundaries. Cartesian cell
fractions use analytical circle/rectangle intersections, not point masks or an
artificial material transition width. The mesh is fixed during differentiation.
"""
from dataclasses import dataclass
import math

import torch

DTYPE = torch.float64


def tensor(value):
    """Convert values to the float64 precision used by the diffusion solvers.

    Parameters
    ----------
    value : array_like or torch.Tensor
        Numerical values to convert without intentionally detaching an existing tensor.

    Returns
    -------
    torch.Tensor
        Float64 tensor with the input's shape.
    """
    return torch.as_tensor(value, dtype=DTYPE)


def _quarter_area(x, y):
    """Integrate the unit disk over signed rectangular limits.

    Parameters
    ----------
    x : torch.Tensor
        Dimensionless signed x limits, broadcast with y.
    y : torch.Tensor
        Dimensionless signed y limits, broadcast with x.

    Returns
    -------
    torch.Tensor
        Signed disk area between the origin and each pair of limits.
    """
    sx, sy = torch.where(x < 0, -1., 1.), torch.where(y < 0, -1., 1.)
    a, b = (sx*x).clamp(max=1.), (sy*y).clamp(max=1.)
    # Finite derivatives at tangencies; the 1e-14 floor changes area only at
    # roundoff and avoids 0*infinity from differentiating sqrt(0).
    ra, rb = (1-a*a).clamp_min(1e-28).sqrt(), (1-b*b).clamp_min(1e-28).sqrt()
    outside = .5*(a*ra+b*rb+torch.atan2(a, ra)+torch.atan2(b, rb)-math.pi/2)
    return sx*sy*torch.where(a*a+b*b <= 1., a*b, outside)


def circle_fractions(x_edges, y_edges, centers, radii):
    """Compute analytical circle intersections with Cartesian cells.

    Parameters
    ----------
    x_edges : torch.Tensor
        Increasing x-cell edges in cm.
    y_edges : torch.Tensor
        Increasing y-cell edges in cm.
    centers : torch.Tensor
        Cylinder centers in cm with shape (n_cylinders, 2).
    radii : torch.Tensor
        Cylinder radii in cm with shape (n_cylinders,).

    Returns
    -------
    torch.Tensor
        Circle area fractions with shape (ny * nx, n_cylinders), in row-major y/x cell
        order.
    """
    x = (x_edges[None, :, None]-centers[:, None, 0:1])/radii[:, None, None]
    y = (y_edges[None, None, :]-centers[:, None, 1:2])/radii[:, None, None]
    primitive = _quarter_area(x, y)*radii[:, None, None]**2
    area = primitive[:, 1:, 1:]-primitive[:, :-1, 1:]-primitive[:, 1:, :-1]+primitive[:, :-1, :-1]
    cell_area = torch.diff(x_edges)[:, None]*torch.diff(y_edges)[None, :]
    return (area/cell_area).permute(2, 1, 0).reshape(-1, len(radii)).clamp(0., 1.)


@dataclass
class Grid:
    """Uniform x/y finite-volume cells extruded through a height in cm.

    Boundaries are ordered x-low, x-high, y-low, y-high. Cylindrical material
    fractions remain differentiable while the grid coordinates stay fixed."""
    lower: tuple
    upper: tuple
    shape: tuple
    boundaries: tuple
    height: float = 1.

    def __post_init__(self):
        """Build the uniform mesh volumes, connections and boundary faces.

        Raises
        ------
        ValueError
            Lateral boundary types, cell counts or extrusion height are invalid.
        """
        if len(self.boundaries) != 4 or any(b not in ('vacuum', 'reflective') for b in self.boundaries):
            raise ValueError('Four lateral vacuum/reflective boundaries are required')
        nx, ny = self.shape
        if min(nx, ny) < 2 or self.height <= 0:
            raise ValueError('Positive height and at least two cells per direction are required')
        self.x_edges = torch.linspace(self.lower[0], self.upper[0], nx+1, dtype=DTYPE)
        self.y_edges = torch.linspace(self.lower[1], self.upper[1], ny+1, dtype=DTYPE)
        self.dx = (self.upper[0]-self.lower[0])/nx
        self.dy = (self.upper[1]-self.lower[1])/ny
        self.volume = self.dx*self.dy*self.height
        ids = torch.arange(nx*ny).reshape(ny, nx)
        self.edge_a = torch.cat((ids[:, :-1].flatten(), ids[:-1, :].flatten()))
        self.edge_b = torch.cat((ids[:, 1:].flatten(), ids[1:, :].flatten()))
        self.edge_area_over_distance = torch.cat((
            torch.full((ny*(nx-1),), self.dy/self.dx*self.height, dtype=DTYPE),
            torch.full(((ny-1)*nx,), self.dx/self.dy*self.height, dtype=DTYPE)))
        self.faces = (ids[:, 0], ids[:, -1], ids[0, :], ids[-1, :])

    def fractions(self, centers, radii):
        """Build validated pin and moderator area fractions on the mesh.

        Parameters
        ----------
        centers : torch.Tensor
            Cylinder centers in cm with shape (n_cylinders, 2).
        radii : torch.Tensor
            Cylinder radii in cm with shape (n_cylinders,).

        Returns
        -------
        torch.Tensor
            Cell-by-region fractions, with one column per pin followed by moderator.

        Raises
        ------
        ValueError
            Pins are invalid, touch or overlap, or fail to lie strictly within the vessel.
        ArithmeticError
            Computed pin fractions overlap beyond numerical tolerance.
        """
        if centers.shape != (len(radii), 2):
            raise ValueError('Expected one x,y center per radius')
        c, r = centers.detach(), radii.detach()
        if bool((r <= 0).any()):
            raise ValueError('Radii must be positive')
        if bool(((c-r[:, None] <= tensor(self.lower)) | (c+r[:, None] >= tensor(self.upper))).any()):
            raise ValueError('Every cylinder must lie strictly within the vessel')
        if len(r) > 1:
            distances = torch.cdist(c, c)+torch.eye(len(r), dtype=DTYPE)*1e30
            if bool((distances <= r[:, None]+r[None, :]).any()):
                raise ValueError('Overlapping or touching pins are unsupported')
        pins = circle_fractions(self.x_edges, self.y_edges, centers, radii)
        water = 1-pins.sum(1, keepdim=True)
        if float(water.detach().min()) < -1e-10:
            raise ArithmeticError('Cell fractions overlap')
        return torch.cat((pins, water.clamp_min(0.)), dim=1)

    def leakage(self, diffusion, extrapolation=1.):
        """Assemble conservative face leakage with Marshak vacuum boundaries.

        Parameters
        ----------
        diffusion : torch.Tensor
            One isotropic diffusion coefficient in cm per cell.
        extrapolation : float or torch.Tensor, optional
            Dimensionless multiplier of the vacuum extrapolation distance.

        Returns
        -------
        torch.Tensor
            Symmetric cell-by-cell leakage matrix for one energy group.
        """
        n = len(diffusion)
        a, b = self.edge_a, self.edge_b
        conductance = 2*diffusion[a]*diffusion[b]/(diffusion[a]+diffusion[b])*self.edge_area_over_distance
        diagonal = torch.zeros(n, dtype=DTYPE).index_add(0, a, conductance).index_add(0, b, conductance)
        off = torch.zeros((n, n), dtype=DTYPE).index_put((a, b), -conductance).index_put((b, a), -conductance)
        for axis, (ids, boundary) in enumerate(zip(self.faces, self.boundaries)):
            if boundary == 'vacuum':
                width, area = (self.dx, self.dy*self.height) if axis < 2 else (self.dy, self.dx*self.height)
                current = area*diffusion[ids]/(.5*width+2*extrapolation*diffusion[ids])
                diagonal = diagonal.index_add(0, ids, current)
        return off+torch.diag(diagonal)


@dataclass
class Materials:
    """Fixed collapsed microscopic data and live atom densities (atom/b-cm).

    Region order is pins, then moderator. Scattering arrays use incoming,
    outgoing group order. No fitted coefficient changes these input tables.
    """
    atom_density: torch.Tensor
    microscopic: dict
    chi: torch.Tensor
    shared_fuel: bool = False

    def __post_init__(self):
        """Enforce shared-fuel composition bindings when requested.

        Raises
        ------
        ValueError
            Fuel regions have different reference densities despite shared_fuel being enabled.
        """
        if self.shared_fuel:
            fuel = self.atom_density[:-1]
            if not torch.allclose(fuel, fuel[:1].expand_as(fuel), rtol=1e-12, atol=0.):
                raise ValueError('Shared fuel requires identical input atom densities in every pin')
            # Retain one physical composition in the graph. Region-specific
            # spectral collapses remain measured data, not independent inputs.
            self.atom_density = torch.cat((fuel[:1].expand_as(fuel), self.atom_density[-1:]))

    @property
    def fuel_parameters(self):
        """Return the number of independent fuel-composition corrections.

        Returns
        -------
        int
            One for shared fuel, otherwise one per fuel region.
        """
        return 1 if self.shared_fuel else len(self.atom_density)-1

    @classmethod
    def from_record(cls, record, shared_fuel=False):
        """Construct material tensors from a native microscopic-data record.

        Parameters
        ----------
        record : dict
            Atom densities, microscopic reaction tables and fission spectrum in region order.
        shared_fuel : bool, optional
            Bind all fuel regions to one composition and calibration coefficient.

        Returns
        -------
        Materials
            Float64 material data with the requested fuel-composition binding.
        """
        return cls(tensor(record['atom_density']),
                   {key: tensor(value) for key, value in record['microscopic'].items()},
                   tensor(record['chi']), shared_fuel)

    def macroscopic(self, log_corrections, density_scale=None, *, capture_scale=1.,
                    scatter_scale=None, diffusion_scale=None, fission_scale=None):
        """Collapse live densities and correction factors into macroscopic data.

        Parameters
        ----------
        log_corrections : torch.Tensor or None
            Log fission corrections for independent fuel compositions; None in solve uses zeros.
        density_scale : torch.Tensor or None, optional
            Dimensionless fuel-density factors followed by one moderator factor; None uses ones.
        capture_scale : float or torch.Tensor, optional
            Dimensionless multiplier broadcast over regional capture cross sections.
        scatter_scale : torch.Tensor or None, optional
            One scattering multiplier per material region; None leaves measured scattering
            unchanged.
        diffusion_scale : torch.Tensor or None, optional
            Region-wise or region-by-group diffusion multipliers; None leaves transport data
            unchanged.
        fission_scale : torch.Tensor or None, optional
            Additional dimensionless multiplier broadcast over region-by-group fission data.

        Returns
        -------
        dict
            Regional reaction tables in cm⁻¹, except kappa_fission in eV/cm, with a
            reaction-consistent total and transport cross section.

        Raises
        ------
        ValueError
            Correction or density-factor shapes violate the fuel binding.
        ArithmeticError
            A corrected transport cross section is nonpositive.
        """
        n = len(self.atom_density)-1
        count = self.fuel_parameters
        if log_corrections.shape != (count,):
            raise ValueError(f'Expected {count} fuel calibration coefficient(s) under the material constraints')
        scale = torch.ones(count+1, dtype=DTYPE) if density_scale is None else density_scale
        if scale.shape != (count+1,):
            raise ValueError(f'Expected {count} fuel density scale(s), followed by one moderator scale')
        correction = log_corrections
        if self.shared_fuel:
            correction = correction.expand(n)
            scale = torch.cat((scale[:1].expand(n), scale[-1:]))
        density = self.atom_density*scale[:, None]
        macro = {key: torch.einsum('mn,mn...->m...', density, value)
                 for key, value in self.microscopic.items()}
        factor = torch.cat((correction.exp(), torch.ones(1, dtype=DTYPE)))[:, None]
        if fission_scale is not None:
            factor = factor*fission_scale
        # Positive fission corrections obey the material bindings. Absorption
        # includes the corrected fission rate; capture and scattering stay fixed.
        macro['absorption'] = capture_scale*(macro['absorption']-macro['fission'])+factor*macro['fission']
        for name in ('fission', 'nu_fission', 'kappa_fission'):
            macro[name] = factor*macro[name]
        if scatter_scale is not None:
            for name in ('scatter', 'nu_scatter'):
                macro[name] = scatter_scale[:, None, None]*macro[name]
            macro['scatter_p1'] = scatter_scale[:, None]*macro['scatter_p1']
        # Use a reaction-consistent total so noisy independent tallies cannot
        # turn cancellation of in-group scattering into spurious absorption.
        macro['total'] = macro['absorption']+macro['scatter'].sum(-1)
        macro['transport'] = macro['total']-macro['scatter_p1']
        if diffusion_scale is not None:
            scale = diffusion_scale[:, None] if diffusion_scale.ndim == 1 else diffusion_scale
            macro['transport'] = macro['transport']/scale
        if bool((macro['transport'].detach() <= 0).any()):
            raise ArithmeticError('Nonpositive collapsed transport cross section')
        return macro


class Diffusion:
    """Solve multigroup diffusion for live cylinder geometry and material data.

    The eigenvalue iteration retains Torch derivatives through geometry,
    densities and correction factors; flux is normalized to total_power in W."""
    def __init__(self, grid, materials, total_power=100., tolerance=2e-11, max_iterations=500):
        """Configure the extruded-cylinder diffusion solve.

        Parameters
        ----------
        grid : Grid
            Fixed Cartesian mesh and lateral boundary conditions.
        materials : Materials
            Fuel and moderator compositions with collapsed microscopic tables.
        total_power : float, optional
            Total fuel power used to normalize pin powers, in W.
        tolerance : float, optional
            Absolute L1 change allowed in the normalized fission source.
        max_iterations : int, optional
            Maximum number of eigenproblem iterations before raising an error.
        """
        self.grid, self.materials, self.total_power = grid, materials, total_power
        self.tolerance, self.max_iterations = tolerance, max_iterations

    def solve(self, centers, radii, log_corrections=None, density_scale=None, *,
              capture_scale=1., scatter_scale=None, diffusion_scale=None,
              vacuum_extrapolation=1., fission_scale=None):
        """Solve the live multigroup cylinder problem and its Torch derivatives.

        Parameters
        ----------
        centers : torch.Tensor
            Cylinder centers in cm with shape (n_cylinders, 2).
        radii : torch.Tensor
            Cylinder radii in cm with shape (n_cylinders,).
        log_corrections : torch.Tensor or None, optional
            Log fission corrections for independent fuel compositions; None in solve uses zeros.
        density_scale : torch.Tensor or None, optional
            Dimensionless fuel-density factors followed by one moderator factor; None uses ones.
        capture_scale : float or torch.Tensor, optional
            Dimensionless multiplier broadcast over regional capture cross sections.
        scatter_scale : torch.Tensor or None, optional
            One scattering multiplier per material region; None leaves measured scattering
            unchanged.
        diffusion_scale : torch.Tensor or None, optional
            Region-wise or region-by-group diffusion multipliers; None leaves transport data
            unchanged.
        vacuum_extrapolation : float or torch.Tensor, optional
            Dimensionless vacuum-extrapolation factor, scalar or one value per energy group.
        fission_scale : torch.Tensor or None, optional
            Additional dimensionless multiplier broadcast over region-by-group fission data.

        Returns
        -------
        dict
            Effective multiplication factor, pin powers in W, unit-production flux, fractions,
            per-source reaction/flux observables, volumes in cm³ and convergence diagnostics.

        Raises
        ------
        RuntimeError
            Fission-source iteration exceeds its iteration limit.
        ArithmeticError
            The solution has nonphysical flux, power or transport data.
        """
        n_pins = len(radii)
        if log_corrections is None:
            log_corrections = torch.zeros(self.materials.fuel_parameters, dtype=DTYPE)
        fractions = self.grid.fractions(centers, radii)
        macro = self.materials.macroscopic(log_corrections, density_scale,
            capture_scale=capture_scale, scatter_scale=scatter_scale,
            diffusion_scale=diffusion_scale, fission_scale=fission_scale)
        total, scatter = fractions@macro['total'], torch.einsum('cm,mgh->cgh', fractions, macro['nu_scatter'])
        diffusion = 1/(3*(fractions@macro['transport']))
        n, groups = total.shape
        volume = self.grid.volume
        rows = []
        extrapolation = torch.as_tensor(vacuum_extrapolation, dtype=total.dtype, device=total.device).expand(groups)
        leakage = [self.grid.leakage(diffusion[:, g], extrapolation[g]) for g in range(groups)]
        for outgoing in range(groups):
            row = []
            for incoming in range(groups):
                block = torch.diag(-scatter[:, incoming, outgoing]*volume)
                if incoming == outgoing:
                    block = block+torch.diag(total[:, outgoing]*volume)+leakage[outgoing]
                row.append(block)
            rows.append(torch.cat(row, dim=1))
        loss = torch.cat(rows, dim=0)
        fission = torch.einsum('cm,mo,mi->coi', fractions, self.materials.chi, macro['nu_fission'])
        lu, pivots = torch.linalg.lu_factor(loss)
        source = torch.ones((n, groups), dtype=DTYPE)/(n*groups*volume)
        for iteration in range(self.max_iterations):
            flux = torch.linalg.lu_solve(lu, pivots, (source*volume).T.reshape(-1, 1)).reshape(groups, n).T
            production = torch.einsum('coi,ci->co', fission, flux)
            keff = production.sum()*volume
            next_source = production/keff
            error = (next_source-source).abs().sum()*volume
            source = next_source
            if float(error.detach()) < self.tolerance:
                break
        else:
            raise RuntimeError(f'Diffusion source iteration failed: residual {float(error.detach()):.3g}')
        flux = flux/keff
        heating = torch.einsum('cm,mg,cg->m', fractions[:, :n_pins], macro['kappa_fission'][:n_pins], flux)*volume
        power = self.total_power*heating/heating.sum()
        residual = loss@flux.T.reshape(-1)-torch.einsum('coi,ci->co', fission, flux).T.reshape(-1)*volume/keff
        relative_residual = torch.linalg.vector_norm(residual)/torch.linalg.vector_norm(loss@flux.T.reshape(-1))
        if not bool(torch.isfinite(power.detach()).all()) or float(flux.detach().min()) < -1e-12:
            raise ArithmeticError('Nonphysical diffusion solution')
        # The original flux has unit integrated fission production. OpenMC
        # tallies use one launched source neutron, whose production is k_eff.
        source_flux = flux*keff
        region_flux = fractions.T@source_flux*volume
        capture = ((macro['absorption']-macro['fission'])*region_flux).sum(-1)
        absorption = (macro['absorption']*region_flux).sum()
        scattering = macro['scatter']*region_flux[:, :, None]
        extra_scatter = ((macro['nu_scatter']-macro['scatter'])*region_flux[:, :, None]).sum()
        escape = torch.stack([operator.sum(0)@source_flux[:, g] for g, operator in enumerate(leakage)])
        return {'keff': keff, 'power_W': power, 'flux': flux, 'fractions': fractions,
                'region_flux_per_source': region_flux, 'capture_per_source': capture,
                'scatter_per_source': scattering, 'leakage_per_source': escape,
                'neutron_balance_error': escape.sum()+absorption-extra_scatter-1,
                'production_per_source': (macro['nu_fission']*region_flux).sum(),
                'iterations': iteration+1, 'eigen_residual': relative_residual,
                'volume_cm3': fractions.sum(0)*volume}
