import torch

from diffusion_surrogate.inventory import InventoryCrossSections
from materials.composition import contract_isotopes, reaction_rates

DTYPE = torch.float64


def test_absorption_fission_scattering_and_depletion_share_isotope_rates():
    """Verify absorption fission scattering and depletion share isotope rates."""
    density = torch.tensor([[2e21, 3e17]], dtype=DTYPE, requires_grad=True)
    fission = torch.tensor([[[.4, 600.], [0., 0.]]], dtype=DTYPE)
    capture = torch.tensor([[[.2, 100.], [2., 2.6e6]]], dtype=DTYPE)
    scattering = torch.tensor([[[[5., 1.], [0., 7.]], [[3., .1], [0., 5.]]]], dtype=DTYPE)
    microscopic = {"absorption": capture + fission, "fission": fission, "scatter": scattering}
    macro = contract_isotopes(density, microscopic)
    flux = torch.tensor([[2e13, 1e13]], dtype=DTYPE)
    xs = torch.stack((capture, fission), dim=2)
    rates = reaction_rates(xs, flux)
    isotope_absorption = (density * rates.sum(-1)).sum()
    diffusion_absorption = (macro["absorption"] * flux).sum()
    torch.testing.assert_close(isotope_absorption, diffusion_absorption)
    torch.testing.assert_close((density * rates[..., 1]).sum(), (macro["fission"] * flux).sum())
    torch.testing.assert_close(macro["scatter"].sum(-1),
                              contract_isotopes(density, {"s": scattering.sum(-1)})["s"])
    gradient = torch.autograd.grad(diffusion_absorption, density)[0]
    torch.testing.assert_close(gradient, ((capture + fission) * flux[:, None] * 1e-24).sum(-1))


def test_live_fuel_removal_leaves_only_explicit_inert_material():
    """Verify live fuel removal leaves only explicit inert material."""
    names = ["U235", "Xe135", "O16", "C0"]
    micro = torch.tensor([[[2., 600.], [1., 2.6e6], [.1, .2], [.01, .02]]], dtype=DTYPE)
    response = {"nuclides": names, "microscopic": {"absorption": micro}}
    table = InventoryCrossSections(response, [{"inert_atom_density": {"C12": .02, "C13": .001}}],
                                    ("U235", "Xe135", "O16", "O18", "C12", "C13"),
                                    aliases={"C12": "C0", "C13": "C0", "O18": "O16"})
    zero = torch.zeros((1, 6), dtype=DTYPE)
    torch.testing.assert_close(table(zero)["absorption"], .021 * micro[:, 3])
    inventory = zero.clone()
    inventory[0, 0], inventory[0, 1] = 2e21, 3e17
    inventory[0, 2], inventory[0, 3] = 4e21, 1e18
    inventory.requires_grad_()
    densities = table.densities(inventory)
    torch.testing.assert_close(densities[:, 2], inventory[:, 2] + inventory[:, 3])
    poisoned = table(inventory)["absorption"]
    clean = table(inventory * torch.tensor([[1., 0., 1., 1., 1., 1.]], dtype=DTYPE))["absorption"]
    torch.testing.assert_close(poisoned - clean, inventory[:, 1, None] * 1e-24 * micro[:, 1])
    assert torch.autograd.grad(poisoned.sum(), inventory)[0][0, 1] > 0
