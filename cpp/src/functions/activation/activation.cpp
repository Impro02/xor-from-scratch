#include <cmath>
#include "functions/activation/activation.h"

Eigen::MatrixXd TanhFunction::activate(const Eigen::MatrixXd &x) const
{
    Eigen::MatrixXd result = x.array().tanh();

    return result;
}

Eigen::MatrixXd TanhFunction::derivative(const Eigen::MatrixXd &x) const
{
    Eigen::MatrixXd result = 1 - x.array().tanh().square();

    return result;
}