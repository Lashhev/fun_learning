#include "neural_layer.hpp"
#include "activation_fuctions.hpp"
#include <random>

using namespace Eigen;
using namespace std;
using namespace fun_learning;

NeuralLayer::NeuralLayer(uint16_t input_number, uint16_t neuron_number, const VectorXd &bias, const Eigen::MatrixXd& weights, std::string activation_func):__weights_number(input_number), __neurons_number(neuron_number)
{
    set_bias(bias);
    set_weights(weights);
    set_activation_func(activation_func);
}


NeuralLayer::NeuralLayer(uint16_t input_number,uint16_t neuron_number,  double min_val, double max_val, std::string activation_func):__weights_number(input_number), __neurons_number(neuron_number)
{
    __bias = VectorXd::Zero(neuron_number);
    __set_random_weights(min_val, max_val);
    set_activation_func(activation_func);
}


void NeuralLayer::set_bias(const Eigen::VectorXd &bias)
{
    __bias = VectorXd(bias);
}

void NeuralLayer::set_activation_func(const std::string &func_type)
{
    __choose_activation_func(func_type);
}


void NeuralLayer::set_weights(const MatrixXd &weights)
{
    __synaptic_weights = MatrixXd(weights);
    __weights_number = __synaptic_weights.rows();
    __neurons_number = __synaptic_weights.cols();
}


void NeuralLayer::get_weights(MatrixXd &weights) const
{
    weights =  MatrixXd(__synaptic_weights);
}

void NeuralLayer::get_biases(VectorXd &bias) const
{
    bias = VectorXd(__bias);
}

std::function<MatrixXd(MatrixXd, double)>& NeuralLayer::get_active_func()
{
    return __activation_func;
}

void NeuralLayer::add_neuron(const Eigen::VectorXd &weights)
{
    MatrixXd new_weights(__weights_number, __neurons_number + 1);
    new_weights << __synaptic_weights, weights;
    __synaptic_weights = MatrixXd(new_weights);
    __neurons_number++;
}
void NeuralLayer::remove(uint16_t n)
{
    MatrixXd new_weights(__weights_number,__neurons_number-1);
    auto diff = (__synaptic_weights.cols()-1) - n;
    MatrixXd M1_1 = __synaptic_weights.block(0,0,__synaptic_weights.rows(),n);
    MatrixXd M1_2 = __synaptic_weights.block(0,n+1,__synaptic_weights.rows(),diff);
    new_weights << M1_1, M1_2;
    __synaptic_weights = MatrixXd(new_weights);
    __neurons_number--;
}
void NeuralLayer::clear()
{
    __synaptic_weights = MatrixXd();
    __neurons_number = 0;
}

void NeuralLayer::__choose_activation_func(const string& func_type)
{
    if(func_type == "sigmoid")
        __activation_func = fun_learning::sigmoid;
    
    else if(func_type=="binary_step")
        __activation_func = fun_learning::binary_step;

    else if(func_type=="tanh_func")
        __activation_func = fun_learning::tanh_func;
    
    else
    {
        __activation_func = fun_learning::sigmoid;
    }
}

void NeuralLayer::__set_random_weights(double min_val, double max_val)
{
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(min_val, max_val);
    //store the random_number in a matrix __synaptic_weights
    __synaptic_weights = Eigen::MatrixXd::Zero(__weights_number, __neurons_number).unaryExpr([&](double dummy){return dist(e2);});
}

MatrixXd fun_learning::operator*(const MatrixXd& input_values, NeuralLayer p)
{
    MatrixXd weights;
    VectorXd bias;
    p.get_weights(weights);
    p.get_biases(bias);
    Eigen::MatrixXd r = (input_values * weights);
    Eigen::MatrixXd b = bias.transpose().replicate(input_values.rows(),1);
    r = r + b;
    auto activation_func = p.get_active_func();
    auto result = activation_func(r, 1);
    return Eigen::MatrixXd(result);
}

// void NeuralLayer::back_propogation(const Eigen::RowVectorXd & input_values, 
//                                         Eigen::RowVectorXd & target_result, 
//                                         double learning_scale, double fval)
// {
//     double e_t;
//     uint32_t cycle_num = 0;
//     do
//     {
//         cycle_num++;
//         Eigen::MatrixXd result = input_values*(*this);
//         e_t = (0.5*(target_result - result).array().square()).sum();
//         std::cout << "cicle_num: " << cycle_num << std::endl << "Neuron precision: " << e_t << std::endl;
//         Eigen::ArrayXXd E = -(target_result-result).array();
//         Eigen::ArrayXXd dR = sigmoid_derivative(result).array();
//         Eigen::ArrayXXd delta = ArrayXXd(E*dR).replicate(input_values.cols(), 1);
//         Eigen::MatrixXd new_weights = __synaptic_weights.array() - learning_scale*delta*input_values.array().transpose().replicate(1, delta.cols());
//         __synaptic_weights = MatrixXd(new_weights);
//     } while (e_t > fval);
//     std::cout << "Training is finished! " << std::endl << "Results:\n" << __synaptic_weights << std::endl;
// }
