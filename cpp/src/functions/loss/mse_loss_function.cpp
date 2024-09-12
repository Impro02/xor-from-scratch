#include "functions/loss/mse_loss_function.h"

double MseLossFunction::loss(const Eigen::MatrixXd &yTrue, const Eigen::MatrixXd &yPred) const
{
    Eigen::VectorXd diff = yTrue - yPred;

    return diff.squaredNorm() / yTrue.size();
}

Eigen::MatrixXd MseLossFunction::derivative(const Eigen::MatrixXd &yTrue, const Eigen::MatrixXd &yPred) const
{
    Eigen::VectorXd diff = yPred - yTrue;

    return 2 * diff / yTrue.size();
}