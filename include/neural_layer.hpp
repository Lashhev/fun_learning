#ifndef __NEURON_LAYER_HPP__
#define __NEURON_LAYER_HPP__

#include "perceptron.hpp"
#include <vector>

namespace fun_learning
{

class Layer
{
public:
    Layer();
    Layer(uint16_t neurons_number, uint16_t inputs_number);

    Layer(uint16_t neurons_number, 
          uint16_t inputs_number, 
          Eigen::VectorXd biases, 
          Eigen::MatrixXd weights);
    void add(const Perceptron &node);
    void insert(uint16_t i, const Perceptron &node);
    void remove(uint16_t i);
    void clear();
    uint16_t size() const;
    void set_weights(const Eigen::MatrixXd& weights);
    void set_biases(const Eigen::VectorXd& biases);
    void get_biases(Eigen::VectorXd& biases) const;
    void get_weights(Eigen::MatrixXd& weights) const;
    Perceptron& operator[](uint16_t key);
private:
    void __create_neurons(uint16_t neurons_number, uint16_t inputs_number);
private:
    std::vector<Perceptron> __neurons;
};

Eigen::MatrixXd operator*(const Eigen::MatrixXd& input_values, Layer l);

} //fun_learning

#endif //__NEURON_LAYER_HPP__