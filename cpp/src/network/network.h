#ifndef NETWORK_H
#define NETWORK_H

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "layer/Layer.h"
#include "functions/loss/loss_function.h"

class Network
{
public:
    // Constructor
    Network(std::shared_ptr<LossFunction> loss);

    // Add layer to network
    void add(std::shared_ptr<Layer> layer);

    // Predict output for given input
    Eigen::MatrixXd predict(const Eigen::MatrixXd &inputData);
    void fit(const Eigen::MatrixXd &xTrain, const Eigen::MatrixXd &yTrain, const int epochs);

private:
    std::vector<std::shared_ptr<Layer>> m_layers;
    std::shared_ptr<LossFunction> m_loss;
};

#endif // NETWORK_H