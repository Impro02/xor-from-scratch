# NUMPY
from numpy.typing import NDArray


class LossFunction:
    def __init__(self) -> None:
        pass

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        raise NotImplementedError

    def prime(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        raise NotImplementedError
