"""Run the agreed steady-state enrichment-shaping study."""

from ..model.optimization import run
from .plotting.render import render


def main():
    """Execute the configured native-verified search after goals are resolved.

    Returns
    -------
    dict
        Completed native evaluations, optimization history and selected design.
    """
    result = run(callback=render)
    return result


if __name__ == "__main__":
    main()
