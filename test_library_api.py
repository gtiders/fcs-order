from ase.build import bulk
from ase.calculators.emt import EMT
from mlfcs.thirdorder import calculate_force_constants
from mlfcs.fourthorder import calculate_force_constants as calc_fc4
import os


def test_api():
    atoms = bulk("Cu", "fcc", a=3.6)
    calc = EMT()

    print("Testing thirdorder API...")
    calculate_force_constants(
        atoms,
        2,
        2,
        2,
        cutoff="-1",
        calculator=calc,
        output_filename="FORCE_CONSTANTS_3RD_TEST",
        verbose=True,
    )

    if os.path.exists("FORCE_CONSTANTS_3RD_TEST"):
        print("Success: FORCE_CONSTANTS_3RD_TEST created")
        # Clean up
        os.remove("FORCE_CONSTANTS_3RD_TEST")
    else:
        print("Failure: FORCE_CONSTANTS_3RD_TEST not created")

    print("\nTesting fourthorder API...")
    calc_fc4(
        atoms,
        2,
        2,
        2,
        cutoff="-1",
        calculator=calc,
        output_filename="FORCE_CONSTANTS_4TH_TEST",
        verbose=True,
    )

    if os.path.exists("FORCE_CONSTANTS_4TH_TEST"):
        print("Success: FORCE_CONSTANTS_4TH_TEST created")
        # Clean up
        os.remove("FORCE_CONSTANTS_4TH_TEST")
    else:
        print("Failure: FORCE_CONSTANTS_4TH_TEST not created")


if __name__ == "__main__":
    test_api()
