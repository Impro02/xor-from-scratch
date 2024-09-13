import numpy as np

from network.network import Network

# LAYERS
from layers import ActivationLayer, FCLayer

# FUNCTIONS
from functions.activation import TanhFunction
from functions.loss import MseLossFunction


def main():
    # training data
    x_train = np.array(
        [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]],
        dtype=np.int32,
    )
    y_train = np.array(
        [[[0]], [[1]], [[1]], [[0]]],
        dtype=np.int32,
    )

    # loss function
    mse_loss = MseLossFunction()

    # network
    network = Network(loss=mse_loss)

    network.add(FCLayer(2, 3, learning_rate=0.05))
    network.add(ActivationLayer(activation=TanhFunction()))
    network.add(FCLayer(3, 1, learning_rate=0.05))
    network.add(ActivationLayer(activation=TanhFunction()))

    # train
    network.fit(
        x_train,
        y_train,
        epochs=1000,
    )

    # test
    out = network.predict(x_train)

    print(out)


if __name__ == "__main__":
    main()
