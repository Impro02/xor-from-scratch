# NUMPY
import numpy as np
from numpy.typing import NDArray

# FUNCTIONS
from functions.activation._base import ActivationFunction


class TanhFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, x: NDArray) -> NDArray:
        return np.tanh(x)

    def prime(self, x: NDArray) -> NDArray:
        return 1 - np.tanh(x) ** 2
