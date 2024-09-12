#ifndef LAYER_H
#define LAYER_H

#include "functions/activation/activation.h"

class Layer
{
public:
    virtual ~Layer() = default;

    virtual Eigen::MatrixXd forwardPropagation(const Eigen::MatrixXd &input) = 0;
    virtual Eigen::MatrixXd backwardPropagation(const Eigen::MatrixXd &outputError) = 0;
};

#endif