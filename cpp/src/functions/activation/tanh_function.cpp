#include <cmath>

#include "functions/activation/tanh_function.h"

Eigen::MatrixXd TanhFunction::activate(const Eigen::MatrixXd &x) const
{
    return x.array().tanh();
}

Eigen::MatrixXd TanhFunction::derivative(const Eigen::MatrixXd &x) const
{
    return 1 - x.array().tanh().square();
}