class MonoPolyError(Exception):
    pass


class MonoPolyInputError(MonoPolyError, ValueError):
    pass


class MonoPolyDesignError(MonoPolyError):
    pass


class InvalidCountMatrixError(MonoPolyInputError):
    pass


class MissingFractionError(MonoPolyInputError):
    pass


class UnpairedFractionError(MonoPolyDesignError):
    pass


class NonEstimableContrastError(MonoPolyDesignError):
    pass


class ModelConvergenceError(MonoPolyError, RuntimeError):
    pass


class MissingOptionalDependencyError(MonoPolyError, ImportError):
    pass