#ifndef __NEURAL_NETWORK_HPP__
#define __NEURAL_NETWORK_HPP__

#include "neural_layer.hpp"
namespace fun_learning
{
class NeuralNetwork
{
public:
    struct Parameters
    {
        uint16_t inputs_number;
        std::vector<uint16_t> nodes_in_layer;
        uint16_t layers_number;
        uint16_t outputs_number;
        std::string activation_func_type;
    };

    NeuralNetwork();
    NeuralNetwork(const Parameters &params);

    void add(const Layer &layer);
    void insert(uint16_t i, const Layer &layer);
};
} // fun_learning
#endif //__NEURAL_NETWORK_HPP__