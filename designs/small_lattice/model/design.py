"""The small lattice's single physical design variable."""

import torch


class Lattice(torch.nn.Module):
    """One bounded common fuel radius; all other geometry settings stay fixed."""

    def __init__(self, settings):
        """Initialize the radius from the design's sole input file.

        Parameters
        ----------
        settings : dict
            Geometry settings with initial radius and bounds in cm.
        """
        super().__init__()
        self.lower, self.upper = settings["radius_bounds_cm"]
        value = (settings["initial_radius_cm"] - self.lower) / (self.upper - self.lower)
        self.coordinate = torch.nn.Parameter(torch.logit(torch.tensor(value, dtype=torch.float64)))

    @property
    def radius_cm(self):
        """Return the bounded, differentiable fuel radius in cm.

        Returns
        -------
        torch.Tensor
            Common fuel radius strictly inside the configured bounds.
        """
        return self.lower + (self.upper - self.lower) * self.coordinate.sigmoid()
