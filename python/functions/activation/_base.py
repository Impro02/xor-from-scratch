# NUMPY
from numpy.typing import NDArray


class ActivationFunction:
    def __init__(self) -> None:
        pass

    def __call__(self, x: NDArray) -> NDArray:
        raise NotImplementedError

    def prime(self, x: NDArray) -> NDArray:
        raise NotImplementedError
