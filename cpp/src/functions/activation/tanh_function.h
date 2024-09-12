#ifndef TANHFUNCTION_H
#define TANHFUNCTION_H

#include <Eigen/Dense>
#include "functions/activation/activation_function.h"

class TanhFunction : public ActivationFunction
{
public:
    virtual ~TanhFunction() = default;

    Eigen::MatrixXd activate(const Eigen::MatrixXd &x) const override;

    Eigen::MatrixXd derivative(const Eigen::MatrixXd &x) const override;
};

#endif