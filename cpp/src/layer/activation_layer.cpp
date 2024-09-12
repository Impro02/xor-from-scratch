#include "layer/activation_layer.h"
#include <memory>

// Constructor
ActivationLayer::ActivationLayer(std::shared_ptr<ActivationFunction> activation)
    : m_activation(activation)
{
}

// Forward Propagation
Eigen::MatrixXd ActivationLayer::forwardPropagation(const Eigen::MatrixXd &x)
{
    m_input = x;

    return m_activation->activate(x);
}

// Backward Propagation
Eigen::MatrixXd ActivationLayer::backwardPropagation(const Eigen::MatrixXd &outputError)
{
    return outputError.cwiseProduct(m_activation->derivative(m_input));
}