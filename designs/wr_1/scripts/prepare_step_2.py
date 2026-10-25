"""Prepare the declared WR-1 step 2 without running neutron transport."""
from coupling.execution import prepare_case
from designs.wr_1.model.case import new_run


def main():
    """Prepare WR-1 scenario 2, export OpenMC XML and print its output path."""
    print(prepare_case(new_run(2)))


if __name__ == "__main__":
    main()
