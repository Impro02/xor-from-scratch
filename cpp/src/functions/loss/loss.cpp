#include "functions/loss/loss.h"

double MseLossFunction::loss(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred) const
{
    Eigen::VectorXd diff = y_true - y_pred;

    return diff.squaredNorm() / y_true.size();
}

Eigen::MatrixXd MseLossFunction::derivative(const Eigen::MatrixXd &y_true, const Eigen::MatrixXd &y_pred) const
{
    Eigen::VectorXd diff = y_pred - y_true;

    return 2 * diff / y_true.size();
}