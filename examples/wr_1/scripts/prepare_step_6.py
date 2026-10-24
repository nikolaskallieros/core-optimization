"""Prepare the declared WR-1 step 6 without running neutron transport."""
from coupling.execution import prepare_case
from examples.wr_1.model.case import new_run


def main():
    """Prepare WR-1 scenario 6, export OpenMC XML and print its output path."""
    print(prepare_case(new_run(6)))


if __name__ == "__main__":
    main()
