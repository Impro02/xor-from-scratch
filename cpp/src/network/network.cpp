#include "network/network.h"
#include <memory>

// Constructor
Network::Network(std::shared_ptr<LossFunction> loss)
    : m_loss(loss)
{
}

// Add layer to network
void Network::add(std::shared_ptr<Layer> layer)
{
    m_layers.push_back(layer);
}

// Predict output for given input
Eigen::MatrixXd Network::predict(const Eigen::MatrixXd &inputData)
{
    Eigen::MatrixXd output = inputData;

    for (auto &layer : m_layers)
    {
        output = layer->forwardPropagation(output);
    }

    return output;
}

// Fit the network to the training data
void Network::fit(const Eigen::MatrixXd &xTrain, const Eigen::MatrixXd &yTrain, const int epochs)
{
    for (int i = 0; i < epochs; i++)
    {
        Eigen::MatrixXd output = predict(xTrain);

        Eigen::MatrixXd error = m_loss->derivative(yTrain, output);

        for (size_t j = m_layers.size() - 1; j >= 0; j--)
        {
            error = m_layers[j]->backwardPropagation(error);
        }
    }
}