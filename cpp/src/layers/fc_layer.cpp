#include <iostream>
#include <memory>

#include "layers/fc_layer.h"

// Constructor
FCLayer::FCLayer(int inputSize, int outputSize, double learningRate)
    : m_learningRate(learningRate),
      m_weights(Eigen::MatrixXd::Random(inputSize, outputSize)),
      m_bias(Eigen::MatrixXd::Random(1, outputSize))
{
}

// Forward Propagation
Eigen::MatrixXd FCLayer::forwardPropagation(const Eigen::MatrixXd &x)
{
    m_input = x;

    m_output = x * m_weights + m_bias;

    return m_output;
}

// Backward Propagation
Eigen::MatrixXd FCLayer::backwardPropagation(const Eigen::MatrixXd &outputError)
{
    auto weightsError = m_input.transpose() * outputError;
    auto biasError = outputError;

    m_weights -= m_learningRate * weightsError;
    m_bias -= m_learningRate * biasError;

    return outputError * m_weights.transpose();
}
