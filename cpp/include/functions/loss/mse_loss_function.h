#ifndef MSELOSSFUNCTION_H
#define MSELOSSFUNCTION_H

#include <Eigen/Dense>
#include "functions/loss/loss_function.h"

class MseLossFunction : public LossFunction
{
public:
    ~MseLossFunction() = default;

    double loss(const Eigen::MatrixXd &yTrue, const Eigen::MatrixXd &yPred) const override;

    Eigen::MatrixXd derivative(const Eigen::MatrixXd &yTrue, const Eigen::MatrixXd &yPred) const override;
};

#endif