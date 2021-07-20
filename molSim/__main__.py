import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        from interfaces import config_reader

        sys.exit(config_reader.main())
    else:
        from interfaces.UI import molSim_ui_main

        sys.exit(molSim_ui_main.main())
