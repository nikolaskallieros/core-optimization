"""WR-1 design variables and their physical parameter mappings."""

import torch
from geometry.channels import ChannelDesign


class Design(ChannelDesign):
    """Select the bare-channel, graphite-pin or graphite-sleeve design variables."""
    def __init__(self, definition, spec):
        """Select and initialize the variables declared by one WR-1 scenario.

        Parameters
        ----------
        definition : dict
            Resolved physical model from ``inputs/model.toml``, supplied by the case adapter.
        spec : dict
            Resolved cycle settings from ``inputs/study.toml`` after scenario overrides.

        Raises
        ------
        ValueError
            The declared parameter names or shapes do not match the selected bare-channel,
            graphite-pin or graphite-sleeve model.
        """
        super().__init__(definition, spec)
        self.register_buffer("fixed_outer_pitch_mm", torch.tensor(spec["fixed_outer_pitch_mm"], dtype=torch.float64))
        expected = {"pitch_mm": (3,), "element_radius_mm": (4,), "enrichment": (4,), "cycle_days": (1,)}
        if "graphite" in spec and spec["graphite"].get("mode") != "assembly_sleeves":
            from .graphite import initialize_graphite
            expected.update(graphite_pitch_mm=(1,), graphite_radius_mm=(4,))
            initialize_graphite(self, spec["graphite"])
        elif "graphite" in spec:
            from .sleeves import initialize_sleeves
            expected.update(graphite_thickness_mm=(21,))
            initialize_sleeves(self, spec["graphite"])
        if {name: tuple(p.shape) for name, p in self.named_parameters()} != expected:
            raise ValueError("WR-1 design parameters do not match the selected optimization step")

    def forward(self):
        """Extend channel geometry with the selected live graphite geometry.

        Returns
        -------
        dict
            Channel and assembly geometry, plus graphite cylinder or sleeve tensors when
            configured. Length-field suffixes specify mm or cm.
        """
        mapped = super().forward()
        if hasattr(self, "graphite_position_index"):
            from .sleeves import sleeve_mapping
            mapped.update(sleeve_mapping(self, mapped))
        elif hasattr(self, "graphite_spec"):
            from .graphite import graphite_mapping
            mapped.update(graphite_mapping(self))
        return mapped

    def ring_pitches(self):
        """Append the fixed passive-ring spacing to the live fuel-ring pitches.

        Returns
        -------
        torch.Tensor
            Four radial spacing increments in mm, including the fixed outer ring.
        """
        # Passive outer sites retain their reference spacing beyond the fuelled rings.
        return torch.cat((self.pitch_mm, self.fixed_outer_pitch_mm))


