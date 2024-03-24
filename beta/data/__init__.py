from .series import BetaSeries
from .dataset import BetaDataset
from .module import BetaDataModule


class R1:
    def __init__(self, a: int) -> None:
        print("R1:", a)


class R2(R1):
    def __init__(self, a: int, b: int) -> None:
        super().__init__(a)
        print("R2:", b)


class R3(R1):
    def __init__(self, a: int, c: float) -> None:
        super().__init__(a)
        print("R3:", c)


class S:
    def __init__(self, r: R1) -> None:
        print("S", type(r))
        print("S", r)
