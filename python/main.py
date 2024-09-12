import numpy as np

from network.network import Network

# LAYERS
from layers import ActivationLayer, FCLayer

# FUNCTIONS
from functions.activation import TanhFunction
from functions.loss import MseLossFunction

# training data
x_train = np.array(
    [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]],
    dtype=np.int32,
)
y_train = np.array(
    [[[0]], [[1]], [[1]], [[0]]],
    dtype=np.int32,
)

# network
net = Network(
    loss=MseLossFunction(),
)
net.add(FCLayer(2, 3))
net.add(ActivationLayer(activation=TanhFunction()))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(activation=TanhFunction()))

# train
net.fit(
    x_train,
    y_train,
    epochs=20000,
    learning_rate=0.05,
)

# test
out = net.predict(x_train)
print(out)
