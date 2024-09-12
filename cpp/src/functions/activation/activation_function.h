#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <Eigen/Dense>

class ActivationFunction
{
public:
    virtual ~ActivationFunction() = default;

    virtual Eigen::MatrixXd activate(const Eigen::MatrixXd &x) const = 0;
    virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd &x) const = 0;
};

#endif // ACTIVATIONFUNCTION_H