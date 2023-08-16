from warnings import warn

from aimsim.exceptions import MordredNotInstalledWarning


def requires_mordred(function):
    try:
        from mordred import Calculator, descriptors

        return function
    except ImportError:
        return MordredNotInstalledWarning(
            "Attempting to call this function ({:s}) requires mordred to be installed from the mordredcommunity package. "
            "Run 'pip install mordredcommunity' or 'conda install -c conda-forge mordredcommunity.".format(
                function.__name__
            )
        )
