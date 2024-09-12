#ifndef ACTIVATIONLAYER_H
#define ACTIVATIONLAYER_H

#include "layer/layer.h"
#include "functions/activation/activation.h"

class ActivationLayer : public Layer
{
public:
    ActivationLayer(std::shared_ptr<ActivationFunction> activation);
    ~ActivationLayer() = default;

    Eigen::MatrixXd forwardPropagation(const Eigen::MatrixXd &input) override;
    Eigen::MatrixXd backwardPropagation(const Eigen::MatrixXd &outputError) override;

private:
    std::shared_ptr<ActivationFunction> m_activation;
    Eigen::MatrixXd m_input;
    Eigen::MatrixXd m_output;
};

#endif