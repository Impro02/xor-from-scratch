# MODULES
from typing import Any, Callable

# NUMPY
from numpy.typing import NDArray

# LAYER
from layers._base import Layer

# FUNCTIONS
from functions.activation import ActivationFunction


# inherit from base class Layer
class ActivationLayer(Layer):

    def __init__(
        self,
        activation: ActivationFunction,
    ) -> None:
        self.activation = activation

    # returns the activated input
    def forward_propagation(self, input_data: NDArray) -> NDArray:
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(
        self,
        output_error: NDArray,
        learning_rate: float,
    ) -> NDArray:
        return self.activation.prime(self.input) * output_error
