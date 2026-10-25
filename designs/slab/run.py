"""Optimize a slab's width using the shared diffusion and optimization interfaces."""
from pathlib import Path
import tomllib

import torch

from diffusion_surrogate import Diffusion3D
from geometry import CadGeometry, MaterialVolume, box
from geometry.diffusion import discretize
from optimization.solvers import minimize
from runtime.paths import RUNS

INPUTS = Path(__file__).with_name("model.toml")
DTYPE = torch.float64


class Slab(torch.nn.Module):
    """One homogeneous slab with vacuum ends and reflective transverse faces."""

    def __init__(self, settings):
        """Create the trainable dimension and fixed manufactured material.

        Parameters
        ----------
        settings : dict
            Parsed model.toml containing geometry and one-group material values.
        """
        super().__init__()
        geometry, material = settings["geometry"], settings["material"]
        self.geometry_settings = geometry
        self.cells = geometry["cells"]
        self.lower, self.upper = geometry["width_bounds_cm"]
        initial = geometry["initial_width_cm"]
        if not 0 < self.lower < initial < self.upper:
            raise ValueError("Initial width must lie strictly inside positive width bounds")
        fraction = torch.tensor((initial - self.lower) / (self.upper - self.lower), dtype=DTYPE)
        self.coordinate = torch.nn.Parameter(torch.logit(fraction))
        absorption = torch.tensor([[material["absorption_per_cm"]]], dtype=DTYPE)
        fission = torch.tensor([[material["fission_per_cm"]]], dtype=DTYPE)
        scatter = torch.tensor([[[material["scatter_per_cm"]]]], dtype=DTYPE)
        self.macro = {
            "absorption": absorption, "fission": fission,
            "nu_fission": material["neutrons_per_fission"] * fission,
            "kappa_fission": material["energy_per_fission_eV"] * fission,
            "scatter": scatter, "nu_scatter": scatter,
            "transport": absorption + scatter.sum(-1), "chi": torch.ones_like(fission),
        }

    @property
    def width_cm(self):
        """Map the unconstrained coordinate into the physical width bounds.

        Returns
        -------
        torch.Tensor
            Differentiable scalar slab width in cm.
        """
        return self.lower + (self.upper - self.lower) * torch.sigmoid(self.coordinate)

    def geometry(self, width=None):
        """Build the single parametric CAD definition for both solver adapters.

        Parameters
        ----------
        width : torch.Tensor or None, optional
            Live width in cm; None evaluates the current design parameter.

        Returns
        -------
        CadGeometry
            Build123d box with the TOML material and face boundary assignments.
        """
        width = self.width_cm if width is None else width
        settings = self.geometry_settings
        upper = torch.stack((width, *width.new_tensor(settings["transverse_size_cm"])))
        domain = box(torch.zeros_like(upper), upper)
        faces = ("x_min", "x_max", "y_min", "y_max", "bottom", "top")
        boundaries = {name: {"type": condition, "slot": i}
                      for i, (name, condition) in enumerate(zip(faces, settings["boundaries"]))}
        material = settings["material"]
        return CadGeometry(domain, (MaterialVolume(material, material, domain),), boundaries, "slab")

    def forward(self):
        """Build the live geometry and solve its one-group diffusion problem.

        Returns
        -------
        dict
            Solver responses including keff, plus the live width_cm tensor.
        """
        width = self.width_cm
        mesh = discretize(self.geometry(width), (self.cells, 1, 1))
        result = Diffusion3D(mesh.grid).solve(mesh.fractions(), self.macro)
        return {**result, "width_cm": width}


def run(inputs=INPUTS):
    """Read one input file, differentiate the geometry and optimize its width.

    Parameters
    ----------
    inputs : str or pathlib.Path, optional
        TOML source for the geometry, manufactured material and optimization goal.

    Returns
    -------
    dict
        Initial/final widths and keff, initial dkeff/dwidth in cm⁻¹, and the
        shared optimizer's convergence and timing record.
    """
    with Path(inputs).open("rb") as stream:
        settings = tomllib.load(stream)
    model = Slab(settings)
    initial = model()
    sensitivity, = torch.autograd.grad(initial["keff"], initial["width_cm"])
    options = settings["optimization"]
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=options["maximum_iterations"],
                                  tolerance_grad=1e-10, tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")
    result = minimize(model, model, lambda fields: (fields["keff"] - options["target_keff"])**2,
                      optimizer=optimizer, keep_best=True)
    with torch.no_grad():
        final = model()
    model.geometry().write_step(RUNS / "slab/geometry.step")
    return {"initial_width_cm": float(initial["width_cm"].detach()),
            "initial_keff": float(initial["keff"].detach()),
            "initial_dkeff_dwidth_per_cm": float(sensitivity),
            "final_width_cm": float(final["width_cm"]), "final_keff": float(final["keff"]),
            "optimization": result}


if __name__ == "__main__":
    result = run()
    print(f"Width: {result['initial_width_cm']:.3f} → {result['final_width_cm']:.3f} cm; "
          f"keff: {result['initial_keff']:.6f} → {result['final_keff']:.6f}")
    print(f"Initial dkeff/dwidth: {result['initial_dkeff_dwidth_per_cm']:.6f} /cm; "
          f"optimization: {result['optimization']['wall_s']:.2f} s")
