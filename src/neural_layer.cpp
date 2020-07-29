#include "neural_layer.hpp"

using namespace Eigen;
using namespace std;
using namespace fun_learning;

Layer::Layer()
{
}

Layer::Layer(uint16_t neurons_number, uint16_t inputs_number)
{
    __create_neurons(neurons_number, inputs_number);  
}

Layer::Layer(uint16_t neurons_number, uint16_t inputs_number, Eigen::VectorXd biases, Eigen::MatrixXd weights)
{
    __create_neurons(neurons_number, inputs_number);  
    set_weights(weights);
    set_biases(biases);
}

void Layer::__create_neurons(uint16_t neurons_number, uint16_t inputs_number)
{
    for(uint16_t i=0; i< neurons_number;i++)
    {
        auto p = Perceptron(inputs_number);
        __neurons.push_back(p) ;
    }
}

void Layer::add(const Perceptron &node)
{
    __neurons.push_back(node);
}

void Layer::insert(uint16_t i, const Perceptron &node)
{
    __neurons.insert(__neurons.begin()+i, node);
}

void Layer::remove(uint16_t i)
{
    __neurons.erase(__neurons.begin()+i);
}

void Layer::clear()
{
    __neurons.clear();
}

void Layer::set_weights(const Eigen::MatrixXd& weights)
{
    for(uint16_t i=0; i< weights.cols();i++)
    {
        Eigen::VectorXd weights_i = weights.col(i);
        auto n = __neurons[i];
        __neurons[i].set_weights(weights_i) ;
    }
}
void Layer::set_biases(const Eigen::VectorXd& biases)
{
    for(uint16_t i=0; i< biases.rows();i++)
    {
        __neurons[i].set_bias(biases[i]) ;
    }
}

void Layer::get_biases(Eigen::VectorXd& biases) const
{
    auto result = VectorXd(__neurons.size());
    for(uint16_t i=0; i<__neurons.size(); i++)
        __neurons[i].get_bias(result[i]);
    biases =  Eigen::VectorXd(result);
}

void Layer::get_weights(Eigen::MatrixXd& weights) const
{
    VectorXd weights0;
    __neurons[0].get_weights(weights0);
    Eigen::MatrixXd result(weights0.rows(), size());
    for(uint16_t i=0; i < size();i++)
    {
        VectorXd weights_i;
        __neurons[i].get_weights(weights_i);
        result.col(i) = weights_i;
    }
    weights =  Eigen::MatrixXd(result);
}

Perceptron& Layer::operator[](uint16_t key)
{
    return __neurons[key];
}

uint16_t Layer::size() const
{
    return __neurons.size();
}

Eigen::MatrixXd fun_learning::operator*(const Eigen::MatrixXd& input_values, Layer l)
{
    Eigen::MatrixXd result(input_values.rows(), l.size());
    for(uint16_t i=0; i < l.size();i++)
    {
        auto x = l[i];
        Eigen::VectorXd r = input_values*x;
        result.col(i) = r;
    }
    return result;
}