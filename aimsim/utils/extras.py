from warnings import warn

from aimsim.exceptions import MordredNotInstalledWarning


def requires_mordred(function):
    try:
        from mordred import Calculator, descriptors

        return function()
    except ImportError:
        return MordredNotInstalledWarning(
            """Attempting to call this function ({:s}) requires mordred to be installed.
            Please use 'pip install aimsim[mordred]' in an environment with the appropriate version of Python.
        """.format(
                function.__name__
            )
        )
