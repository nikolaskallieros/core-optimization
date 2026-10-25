"""Prepare the uniform fresh-fuel reference for enrichment shaping."""

from ..model.optimization import run
from ..model.case import settings
from ..model.reference import prepare_reference
from .plotting.render import render


def main():
    """Retain the uniform reference at the fixed TOML geometry.

    Returns
    -------
    dict
        Prepared geometry, uniform native reference and its saved run directory.
    """
    reference = prepare_reference() if settings()["optimization"]["keff_reference"] == "critical" else None
    result = run(reference, prepare_only=True)
    render(result)
    return result


if __name__ == "__main__":
    main()
