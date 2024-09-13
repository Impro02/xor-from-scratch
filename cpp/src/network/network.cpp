#include <iostream>
#include <memory>

#include "network/network.h"

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

    Eigen::MatrixXd result(inputData.rows(), 1);

    for (int i = 0; i < inputData.rows(); i++)
    {
        Eigen::MatrixXd output = inputData.row(i);
        for (auto &layer : m_layers)
        {
            output = layer->forwardPropagation(output);
        }
        result.row(i) = output;
    }

    return result;
}

// Fit the network to the training data
void Network::fit(const Eigen::MatrixXd &xTrain, const Eigen::MatrixXd &yTrain, const int epochs)
{
    for (int i = 0; i < epochs; i++)
    {
        double err = 0;
        for (int j = 0; j < xTrain.rows(); j++)
        {
            Eigen::MatrixXd output = xTrain.row(j);
            for (auto &layer : m_layers)
            {
                output = layer->forwardPropagation(output);
            }

            err += m_loss->loss(yTrain.col(j), output);

            auto error = m_loss->derivative(yTrain.col(j), output);
            for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
            {
                auto &layer = *it;
                error = layer->backwardPropagation(error);
            }
        }

        if ((i + 1) % 100 == 0)
        {
            std::cout << "Epoch: " << i + 1 << " error: " << err << std::endl;
        }
    }
}