# MODULES
from typing import Any, List

# NUMPY
from numpy.typing import NDArray

# LAYER
from layers._base import Layer

# FUNCTIONS
from functions.loss import LossFunction


class Network:

    def __init__(
        self,
        loss: LossFunction,
    ) -> None:
        self._layers: List[Layer] = []
        self._loss = loss

    # add layer to network
    def add(self, layer: Layer) -> None:
        self._layers.append(layer)

    # predict output for given input
    def predict(self, input_data: NDArray[Any]):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self._layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(
        self,
        x_train: NDArray[Any],
        y_train: NDArray[Any],
        epochs: int,
    ) -> None:
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err: float = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self._layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self._loss(y_train[j], output)

                # backward propagation
                error = self._loss.prime(y_train[j], output)
                for layer in reversed(self._layers):
                    error = layer.backward_propagation(
                        output_error=error,
                    )

            # calculate average error on all samples
            err /= samples

            if i == 0 or (i + 1) % 100 == 0:
                print("epoch %d/%d   error=%f" % (i + 1, epochs, err))
