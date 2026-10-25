"""Run and plot the small lattice's complete local optimization workflow."""

from ..model.case import run
from .plotting.render import render


def main():
    """Execute the configured case and render its verified native/surrogate results."""
    result = run()
    render(result)


if __name__ == "__main__":
    main()
