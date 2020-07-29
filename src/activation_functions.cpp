#include "activation_fuctions.hpp"

using namespace Eigen;

VectorXd fun_learning::sigmoid(const VectorXd & x, double a)
{
    VectorXd result = (1 / (1 + (-x*a).array().exp())).matrix();
        return result;
}

VectorXd fun_learning::tanh_func(const VectorXd &  x, double a)
{
    VectorXd result =  2*sigmoid(2*x, a).array() - 1;
    return result;
}

VectorXd fun_learning::binary_step(const VectorXd &  x, double threshold)
{
    VectorXd result(1);
    if(x(0) < threshold)
    {
        result << 0;
        return result;
    } 
    else
    {
        result << 1;
        return result;
    }
        
}