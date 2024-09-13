#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include <Eigen/Dense>

class LossFunction
{
public:
    virtual ~LossFunction() = default;

    virtual double loss(const Eigen::MatrixXd &yTrue, const Eigen::MatrixXd &yPred) const = 0;
    virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd &yTrue, const Eigen::MatrixXd &yPred) const = 0;
};

#endif