"""Command line and GUI access point."""
import sys


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
