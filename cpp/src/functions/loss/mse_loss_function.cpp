#include "functions/loss/mse_loss_function.h"

double MseLossFunction::loss(const Eigen::MatrixXd &yTrue, const Eigen::MatrixXd &yPred) const
{
    auto diff = yTrue - yPred;

    return diff.squaredNorm() / yTrue.size();
}

Eigen::MatrixXd MseLossFunction::derivative(const Eigen::MatrixXd &yTrue, const Eigen::MatrixXd &yPred) const
{
    auto diff = yPred - yTrue;

    return 2 * diff / yTrue.size();
}