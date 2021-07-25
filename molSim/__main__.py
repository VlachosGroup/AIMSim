"""Command line and GUI access point.

Raises:
    MissingRDKitError: Raised if RDKit installation is missing.
"""
import sys

from molSim.exceptions import MissingRDKitError

try:
    from rdkit import Chem
except ImportError:
    raise MissingRDKitError(
        "RDKit installation not found! Run `conda install -c rdkit rdkit`."
    )


def start_molSim():
    """Opens the GUI if no commands are given, otherwise passes through config
    to the configuration reader.
    """
    if len(sys.argv) > 1:
        from interfaces import config_reader

        sys.exit(config_reader.main())
    else:
        from interfaces.UI import molSim_ui_main

        sys.exit(molSim_ui_main.main())


if __name__ == "__main__":
    start_molSim()
