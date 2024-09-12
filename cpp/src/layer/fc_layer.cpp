#include "layer/fc_layer.h"
#include <memory>

// Constructor
FCLayer::FCLayer(int inputSize, int outputSize, double learningRate)
    : m_learningRate(learningRate),
      m_weights(Eigen::MatrixXd::Random(outputSize, inputSize)),
      m_bias(Eigen::MatrixXd::Random(outputSize, 1))
{
}

// Forward Propagation
Eigen::MatrixXd FCLayer::forwardPropagation(const Eigen::MatrixXd &x)
{
    m_input = x;

    m_output = m_weights * x + m_bias;

    return m_output;
}

// Backward Propagation
Eigen::MatrixXd FCLayer::backwardPropagation(const Eigen::MatrixXd &outputError)
{
    Eigen::MatrixXd weightsError = outputError * m_input.transpose();
    Eigen::MatrixXd biasError = outputError;

    m_weights -= m_learningRate * weightsError;
    m_bias -= m_learningRate * biasError;

    return m_weights.transpose() * outputError;
}
