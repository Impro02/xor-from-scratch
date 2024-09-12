#ifndef FCLAYER_H
#define FCLAYER_H

#include "layer/layer.h"

class FCLayer : public Layer
{
public:
    FCLayer(int inputSize, int outputSize, double learningRate);
    ~FCLayer() = default;

    Eigen::MatrixXd forwardPropagation(const Eigen::MatrixXd &input) override;
    Eigen::MatrixXd backwardPropagation(const Eigen::MatrixXd &outputError) override;

private:
    double m_learningRate;
    Eigen::MatrixXd m_input;
    Eigen::MatrixXd m_output;
    Eigen::MatrixXd m_weights;
    Eigen::MatrixXd m_bias;
};

#endif