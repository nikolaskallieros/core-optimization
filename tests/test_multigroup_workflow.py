"""Independent references for native group ordering and the small lattice history."""

import math
import tomllib

import pytest
import torch

from coupling.multigroup import MultigroupTransport
from designs.small_lattice.build_geometry import build_geometry
from designs.small_lattice.model.design import Lattice
from designs.small_lattice.model.materials import Materials, build_chain
from designs.small_lattice.model.case import INPUTS, make_history
from geometry import CadGeometry, MaterialVolume, box
from geometry.diffusion import discretize


@pytest.fixture
def lattice():
    """Bind a small independent mesh to the real lattice/depletion interfaces.

    Returns
    -------
    tuple
        Settings, live design, material mapping and differentiable history.
    """
    settings = tomllib.loads(INPUTS.read_text())
    design = Lattice(settings["geometry"])
    materials = Materials(settings["materials"], 9)
    chain = build_chain(settings["chain"], settings["materials"])
    mesh = discretize(lambda: build_geometry(settings["geometry"], design.radius_cm), (12, 12, 1))
    model = make_history(settings, mesh, materials, chain)
    return settings, design, mesh, model


def test_lattice_material_volumes_and_radius_derivative(lattice):
    """Recover analytic cylinder volumes and their derivative through mesh cuts.

    Parameters
    ----------
    lattice : tuple
        Live lattice fixture.
    """
    settings, design, mesh, _ = lattice
    volumes = mesh.fractions().T @ mesh.grid.volumes
    radius = design.radius_cm
    expected = math.pi * radius**2 * settings["geometry"]["height_cm"]
    torch.testing.assert_close(volumes[:9], expected.expand(9), rtol=1e-11, atol=1e-11)
    total = math.prod(settings["geometry"]["shape"]) * math.prod(settings["geometry"]["pitch_cm"])
    total *= settings["geometry"]["height_cm"]
    assert float(volumes.sum().detach()) == pytest.approx(total, rel=1e-12)
    actual, = torch.autograd.grad(volumes[:9].sum(), design.coordinate, retain_graph=True)
    analytic, = torch.autograd.grad(9 * expected, design.coordinate)
    torch.testing.assert_close(actual, analytic, rtol=1e-10, atol=1e-10)


def test_lattice_history_keeps_geometry_and_depletion_gradients(lattice):
    """Compare a complete history derivative with an independent finite difference.

    Parameters
    ----------
    lattice : tuple
        Live lattice fixture.
    """
    settings, design, _, model = lattice
    output = model()
    derivative, = torch.autograd.grad(output["keff"], design.coordinate)
    original = design.coordinate.detach().clone()
    step = settings["verification"]["gradient_coordinate_step"]
    with torch.no_grad():
        design.coordinate.copy_(original + step)
        upper = model()["keff"]
        design.coordinate.copy_(original - step)
        lower = model()["keff"]
        design.coordinate.copy_(original)
    torch.testing.assert_close(derivative, (upper - lower) / (2 * step), rtol=2e-4, atol=1e-8)
    assert bool((output["inventory"][:, 0] < output["inventory_history"][0, :, 0]).all())
    assert bool((output["inventory"][:, 1:] > 0).all())
    expected = settings["history"]["power_W"] * sum(settings["history"]["durations_days"]) * 86400
    assert float(output["energy_j"].detach().sum()) == pytest.approx(expected, rel=1e-12)
    assert [name for name, _ in model.named_parameters()] == ["response.transport.log_factors"]


def test_zero_history_recovers_fresh_transport(lattice):
    """Removing the depletion intervals recovers the initial transport state.

    Parameters
    ----------
    lattice : tuple
        Live lattice fixture.
    """
    _, _, _, model = lattice
    state = model.initial_state()
    expected = model.response(state.inventory)
    model.durations_days = torch.empty(0, dtype=torch.float64)
    actual = model()
    torch.testing.assert_close(actual["keff"], expected["keff"])
    torch.testing.assert_close(actual["inventory"], state.inventory, rtol=0, atol=0)
    assert actual["energy_j"].count_nonzero() == 0
    assert actual["time_days"].tolist() == [0.]


def test_native_two_group_infinite_medium_preserves_group_order_and_power(tmp_path):
    """Recover an analytical two-group eigenpair through CAD and actual OpenMC.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated native workspace.
    """
    domain = box([-1., -1., -1.], [1., 1., 1.])
    names = ("x_min", "x_max", "y_min", "y_max", "bottom", "top")
    boundaries = {name: {"type": "reflective", "slot": i} for i, name in enumerate(names)}
    cad = CadGeometry(domain, (MaterialVolume("fuel", "fuel", domain),), boundaries)
    absorption = torch.tensor([[.05, .10]], dtype=torch.float64)
    scatter = torch.tensor([[[.4, .15], [0., .9]]], dtype=torch.float64)
    fission = torch.tensor([[.02, .06]], dtype=torch.float64)
    macro = {"absorption": absorption, "scatter": scatter, "nu_scatter": scatter,
             "fission": fission, "nu_fission": 2.5 * fission, "kappa_fission": 2e8 * fission,
             "transport": absorption + scatter.sum(-1), "chi": torch.tensor([[1., 0.]], dtype=torch.float64)}
    native = MultigroupTransport(cad, tmp_path / "native", [0., 1e5, 20e6],
        {"particles": 2000, "batches": 40, "inactive": 10, "threads": 1, "seed": 39271}, power_W=100.)
    result = native(macro)
    # Thermal balance gives phi_thermal/phi_fast = 0.15/0.10; fast balance
    # then gives k = (0.05 + 0.15*1.5) / (0.05 + 0.15) = 1.375.
    assert abs(float(result["keff"]) - 1.375) < max(6 * float(result["standard_error"]), .02)
    ratio = float(result["region_flux_per_source"][0, 1] / result["region_flux_per_source"][0, 0])
    assert ratio == pytest.approx(1.5, rel=.03)
    integrated_heating = (macro["kappa_fission"] * result["region_flux_per_source"]).sum()
    recovered_power = integrated_heating * result["source_rate_per_s"] * 1.602176634e-19
    assert float(recovered_power) == pytest.approx(100., rel=1e-10)
    assert float(result["volume_cm3"][0]) == pytest.approx(8., rel=1e-12)
