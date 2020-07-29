#include "perceptron.hpp"
#include "activation_fuctions.hpp"

using namespace Eigen;
using namespace std;

Perceptron::Perceptron(uint16_t input_number, double bias, const Eigen::VectorXd& weights, std::string activation_func):__weights_number(input_number)
{
    set_bias(bias);
    set_weights(weights);
    set_activation_func(activation_func);
}


Perceptron::Perceptron(uint16_t input_number, double min_val, double max_val):__weights_number(input_number)
{
    __synaptic_weights = VectorXd(input_number);
    __set_random_weights(input_number,min_val, max_val);
    set_activation_func("sigmoid");
}


void Perceptron::set_bias(double bias)
{
    __bias = bias;
}

void Perceptron::set_activation_func(const std::string &func_type)
{
    __choose_activation_func(func_type);
}


void Perceptron::set_weights(const Eigen::VectorXd &weights)
{
    __synaptic_weights = Eigen::VectorXd(weights);
    __weights_number = __synaptic_weights.rows();
}


void Perceptron::get_weights(Eigen::VectorXd &weights) const
{
    weights =  VectorXd(__synaptic_weights);
}

void Perceptron::get_bias(double& bias) const
{
    bias = __bias;
}

std::function<MatrixXd(MatrixXd, double)>& Perceptron::get_active_func()
{
    return __activation_func;
}

void Perceptron::__choose_activation_func(const string& func_type)
{
    if(func_type == "sigmoid")
        __activation_func = sigmoid;
    
    else if(func_type=="binary_step")
        __activation_func = binary_step;

    else if(func_type=="tanh_func")
        __activation_func = tanh_func;
    
    else
    {
        __activation_func = sigmoid;
    }
}

void Perceptron::__set_random_weights(uint16_t weight_number, double min_val, double max_val)
{
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-5, 5);
    //store the random_number in a matrix __synaptic_weights
    __synaptic_weights = Eigen::VectorXd::Zero(weight_number).unaryExpr([&](double dummy){return dist(e2);});
}

Eigen::VectorXd operator*(const MatrixXd& input_values, Perceptron p)
{
    Eigen::VectorXd weights;
    double bias;
    p.get_weights(weights);
    p.get_bias(bias);
    Eigen::VectorXd r = (input_values * weights).array() + bias;
    auto activation_func = p.get_active_func();
    auto result = activation_func(r, 1);
    return Eigen::VectorXd::Map(result.data(), result.rows(), result.cols());
}
