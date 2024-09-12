#include <iostream>
#include <Eigen/Dense>
#include "functions/loss/loss.h"
#include "layer/activation_layer.h"
#include "layer/fc_layer.h"
#include "network/network.h"

int main()
{
    Eigen::MatrixXd x_train;
    Eigen::MatrixXd y_train;

    x_train << 0, 0,
        0, 1,
        1, 0,
        1, 1;
    y_train << 0,
        1,
        1,
        0;

    std::shared_ptr<MseLossFunction> mseLossFunction = std::make_shared<MseLossFunction>();
    Network network = Network(mseLossFunction);

    network.add(std::make_shared<FCLayer>(2, 3, 0.05));
    network.add(std::make_shared<ActivationLayer>(std::make_shared<TanhFunction>()));
    network.add(std::make_shared<FCLayer>(3, 1, 0.05));
    network.add(std::make_shared<ActivationLayer>(std::make_shared<TanhFunction>()));

    network.fit(x_train, y_train, 1000);

    Eigen::MatrixXd output = network.predict(x_train);

    std::cout << output << std::endl;

    return 0;
}