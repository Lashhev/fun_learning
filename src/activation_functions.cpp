#include "activation_fuctions.hpp"
#include "version_info.h"

using namespace Eigen;

MatrixXd fun_learning::sigmoid(const MatrixXd & x, double a)
{
    MatrixXd result = (1 / (1 + (-x*a).array().exp())).matrix();
        return result;
}

MatrixXd fun_learning::tanh_func(const MatrixXd &  x, double a)
{
    MatrixXd result =  2*sigmoid(2*x, a).array() - 1;
    return result;
}

MatrixXd fun_learning::binary_step(const MatrixXd &  x, double threshold)
{
    return MatrixXd((x.array() >= threshold).cast<double>());
}

Eigen::MatrixXd fun_learning::sigmoid_derivative(const Eigen::MatrixXd &  x)
{
    return x.array()*(1-x.array());
}