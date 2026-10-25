"""Geometry-independent material calibration with an explicit solver interface."""
import torch

from .solver import tensor


class CalibratedDiffusion(torch.nn.Module):
    """Fit material coefficients while the caller owns geometry and discretization.

    Every basis has leading parameter dimension followed by region/group axes;
    no region is implicitly fuel, coolant, reflector or absorber. The same
    correction applies to all regions bound to the same caller-defined factor.
    The solver accepts regional fractions and corrected macroscopic tables;
    calibration does not construct, simplify or optimize the geometry.
    """
    def __init__(self, solver, geometry, macro, bases, parameter_names, *, power_W,
                 initial=None, responses=None, equivalence=None):
        """Bind measured material data to a solver and caller-defined geometry.

        Parameters
        ----------
        solver : object
            Solver exposing ``solve(fractions, macro, power_W=..., extrapolation=...)``.
            Cartesian, triangular-prism and other compatible meshes use the same interface.
        geometry : callable
            Zero-argument callback returning live cell-by-region volume fractions in the
            material-table order. Its design parameters remain owned by the caller.
        macro : dict
            Measured region/group macroscopic tables in cm⁻¹, kappa_fission in eV/cm and
            dimensionless chi.
        bases : dict
            Log-correction bases with a leading parameter axis and trailing region/group or
            boundary/group axes.
        parameter_names : sequence of str
            Names corresponding to the leading axis of each correction basis.
        power_W : float
            Total power in W used to normalize predicted powers.
        initial : array_like or None, optional
            Initial log correction factors; None starts at zero, corresponding to unit factors.
        responses : callable or None, optional
            Map solver outputs to additional comparison fields. None retains the solver's
            regional responses without assuming spatial axes or a comparison mesh.
        equivalence : array_like or None, optional
            Parameter-by-region-by-group basis for positive SPH equivalence factors.
            These scale incoming-group reaction coefficients and reconstructed physical
            flux together; None disables this additional correction.

        Raises
        ------
        ValueError
            A correction basis is unknown or has a different parameter count from the names.

        Notes
        -----
        Fission, capture and scattering factors correct macroscopic reaction data. Diffusion
        factors act through the transport cross section. Boundary extrapolation changes only
        when a boundary basis is explicitly supplied. Fitting updates log_factors; geometry
        callbacks may retain live design derivatives for subsequent optimization.
        """
        super().__init__()
        self.solver = solver
        # A geometry module can carry a live design without becoming a fitted child module.
        object.__setattr__(self, 'geometry', geometry)
        object.__setattr__(self, 'responses', responses)
        self.parameter_names = tuple(parameter_names)
        self.power_W = power_W
        self._macro_names, self._basis_names = tuple(macro), tuple(bases)
        if equivalence is not None:
            value = tensor(equivalence)
            if value.shape != (len(parameter_names), *tensor(macro['absorption']).shape):
                raise ValueError('Equivalence basis must match parameters and region/group axes')
            self.register_buffer('equivalence_basis', value.clone())
        unknown = set(bases) - {'capture', 'fission', 'scatter', 'diffusion', 'boundary'}
        if unknown:
            raise ValueError(f'Unknown correction bases: {sorted(unknown)}')
        for name, value in macro.items():
            self.register_buffer('macro_'+name, tensor(value).clone())
        for name, value in bases.items():
            value = tensor(value)
            if value.shape[0] != len(parameter_names):
                raise ValueError('Correction bases must match named parameters')
            self.register_buffer('basis_'+name, value.clone())
        self.log_factors = torch.nn.Parameter(tensor([0.]*len(parameter_names) if initial is None else initial))

    def macroscopic(self, material_delta=None):
        """Apply live material changes before positive fitted corrections.

        Parameters
        ----------
        material_delta : dict or None, optional
            Live changes to measured macroscopic tables, with identical region/group axes and
            units; None leaves the physical reference unchanged.

        Returns
        -------
        tuple
            Corrected macroscopic tables and a scalar or face-by-group vacuum-extrapolation
            multiplier.

        Raises
        ------
        ValueError
            A material-delta table has incompatible axes.

        Notes
        -----
        Deltas have the same region/group axes as the measured tables. Their construction
        (number densities, concentrations, etc.) belongs to the model adapter, and their Torch
        graph is preserved here.
        """
        macro = {name: getattr(self, 'macro_'+name) for name in self._macro_names}
        for name, change in (material_delta or {}).items():
            if change.shape != macro[name].shape:
                raise ValueError(f'Mismatched material axes: {name}')
            macro[name] = macro[name]+change
        moment = macro.get('scatter_p1')
        if moment is None:
            moment = macro['absorption']+macro['scatter'].sum(-1)-macro['transport']

        def scale(name):
            """Evaluate the positive factor represented by one correction basis.

            Parameters
            ----------
            name : str
                Basis identity such as capture, fission, scatter, diffusion or boundary.

            Returns
            -------
            float or torch.Tensor
                Exponentiated basis contraction, or unity when that basis is absent.
            """
            if name not in self._basis_names:
                return 1.
            return torch.einsum('p,p...->...', self.log_factors, getattr(self, 'basis_'+name)).exp()

        capture = (macro['absorption']-macro['fission'])*scale('capture')
        fission_factor = scale('fission')
        for name in ('fission', 'nu_fission', 'kappa_fission'):
            macro[name] = macro[name]*fission_factor
        macro['absorption'] = capture+macro['fission']
        scattering = scale('scatter')
        for name in ('scatter', 'nu_scatter'):
            macro[name] = macro[name]*(scattering if isinstance(scattering, float) else scattering[..., None])
        macro['transport'] = (macro['absorption']+macro['scatter'].sum(-1)-moment*scattering)/scale('diffusion')
        return macro, scale('boundary')

    def forward(self, material_delta=None):
        """Apply material corrections and solve the explicitly supplied geometry.

        Parameters
        ----------
        material_delta : dict or None, optional
            Live changes to measured macroscopic tables, with identical region/group axes and
            units; None leaves the physical reference unchanged.

        Returns
        -------
        dict
            Solver observables and any explicitly requested comparison responses. SPH
            equivalence exposes both physical flux and the uncorrected diffusion amplitude.
        """
        fractions = getattr(self, '_live_fractions', None)
        if fractions is None:
            fractions = self.geometry()
        macro, boundary = self.macroscopic(material_delta)
        equivalence = None
        if hasattr(self, 'equivalence_basis'):
            equivalence = torch.einsum('p,pmg->mg', self.log_factors, self.equivalence_basis).exp()
            # SPH scales incoming-group reactions, diffusion coefficients and
            # reconstructed flux together, preserving measured reaction ratios.
            for name in ('absorption', 'fission', 'nu_fission', 'kappa_fission', 'total_tracklength', 'scatter_p1'):
                if name in macro:
                    macro[name] = macro[name] * equivalence
            for name in ('scatter', 'nu_scatter'):
                macro[name] = macro[name] * equivalence[..., None]
            macro['transport'] = macro['transport'] / equivalence
        result = self.solver.solve(fractions, macro, power_W=self.power_W, extrapolation=boundary)
        if equivalence is not None:
            result['diffusion_amplitude_per_source'] = result['flux_per_source']
            result['region_flux_per_source'] = result['region_flux_per_source'] * equivalence
            result['flux_per_source'] = result['flux_per_source'] * (fractions @ equivalence).reshape_as(result['flux_per_source'])
            result['equivalence_factors'] = equivalence
        if self.responses is not None:
            result.update(self.responses(result))
        return result
