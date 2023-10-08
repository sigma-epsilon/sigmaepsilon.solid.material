import warnings


class SigmaEpsilonMaterialWarning(UserWarning):
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return f"[sigmaepsilon.solid.core] {self.args[0]}"
    

warnings.simplefilter("always", SigmaEpsilonMaterialWarning)