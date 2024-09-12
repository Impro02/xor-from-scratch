#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include <Eigen/Dense>

class LossFunction
{
public:
    virtual ~LossFunction() = default;

    virtual double loss(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred) const = 0;
    virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred) const = 0;
};

class MseLossFunction : public LossFunction
{
public:
    ~MseLossFunction() = default;

    double loss(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred) const override;

    Eigen::MatrixXd derivative(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred) const override;
};

#endif // LOSSFUNCTION_H