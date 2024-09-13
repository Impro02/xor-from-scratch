#include <iostream>
#include <Eigen/Dense>

#include "functions/loss/mse_loss_function.h"
#include "functions/activation/tanh_function.h"
#include "layers/activation_layer.h"
#include "layers/fc_layer.h"
#include "network/network.h"

int main()
{
    Eigen::MatrixXd xTrain(4, 2);
    Eigen::MatrixXd yTrain(1, 4);

    xTrain << 0, 0,
        0, 1,
        1, 0,
        1, 1;
    yTrain << 0, 1, 1, 0;

    std::shared_ptr<MseLossFunction> mseLossFunction = std::make_shared<MseLossFunction>();
    Network network = Network(mseLossFunction);

    network.add(std::make_shared<FCLayer>(2, 3, 0.05));
    network.add(std::make_shared<ActivationLayer>(std::make_shared<TanhFunction>()));
    network.add(std::make_shared<FCLayer>(3, 1, 0.05));
    network.add(std::make_shared<ActivationLayer>(std::make_shared<TanhFunction>()));

    network.fit(xTrain, yTrain, 1000);

    Eigen::MatrixXd output = network.predict(xTrain);

    std::cout << "output:\n"
              << output << std::endl;

    return 0;
}