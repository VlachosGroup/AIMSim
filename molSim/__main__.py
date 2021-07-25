import sys

try:
    from rdkit import Chem
except ImportError:
    raise BaseException(
        "RDKit installation not found! Run `conda install -c rdkit rdkit`."
    )
    pass


def start_molSim():
    if len(sys.argv) > 1:
        from interfaces import config_reader

        sys.exit(config_reader.main())
    else:
        from interfaces.UI import molSim_ui_main

        sys.exit(molSim_ui_main.main())


if __name__ == "__main__":
    start_molSim()
