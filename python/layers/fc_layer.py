# Numpy
import numpy as np
from numpy.typing import NDArray

# LAYER
from layers._base import Layer


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of outpout neurons
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float,
    ) -> None:
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.learning_rate = learning_rate

    # returns for a given input
    def forward_propagation(self, input_data: NDArray) -> NDArray:
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error =dE/dY. Returns input_error=dE/dW.
    def backward_propagation(
        self,
        output_error: NDArray,
    ) -> NDArray:
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= self.learning_rate * weights_error
        self.bias -= self.learning_rate * output_error
        return input_error
