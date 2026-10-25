"""Run one WR-1 scenario using the shared native/surrogate cycle engine."""
from coupling.execution import run_case
from designs.wr_1.model.case import new_run

STEP = 1


def main():
    """Run the scenario selected by STEP through the shared optimization engine."""
    run_case(new_run(STEP))


if __name__ == "__main__":
    main()
