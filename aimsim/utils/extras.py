def requries_mordred(function):
    try:
        from mordred import Calculator, descriptors

        function()
    except ImportError:
        raise RuntimeError(
            """Attempting to call this function ({:s}) requires mordred to be installed.
            Please use 'pip install aimsim[mordred]' in an environment with the appropriate version of Python.
        """.format(
                function.__name__
            )
        )
