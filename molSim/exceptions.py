class NotInitializedError(AttributeError):
    """This is used when a class is called without initialization."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class MordredCalculatorError(RuntimeError):
    """Used in descriptor.py when the Mordred property calculator fails."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)


class MissingRDKitError(RuntimeError):
    """Used when attempting to start molSim without RDKit installed."""

    def __init__(self, message=None):
        self.message = message
        super().__init__(message)
