# NUMPY
import numpy as np
from numpy.typing import NDArray

# FUNCTIONS
from functions.loss._base import LossFunction


class MseLossFunction(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true: NDArray, y_pred: NDArray) -> float:
        return np.mean(np.power(y_true - y_pred, 2))

    def prime(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return 2 * (y_pred - y_true) / y_true.size
