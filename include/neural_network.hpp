#ifndef __NEURAL_NETWORK_HPP__
#define __NEURAL_NETWORK_HPP__

#include "neural_layer.hpp"
#include <vector>
namespace fun_learning
{
class NeuralNetwork
{
public:
    // struct Parameters
    // {
    //     uint16_t inputs_number;
    //     std::vector<uint16_t> nodes_in_layer;
    //     uint16_t layers_number;
    //     uint16_t outputs_number;
    //     std::string activation_func_type;
    // };

    NeuralNetwork();
    // NeuralNetwork(const Parameters &params);

    void feed_forward(const Eigen::MatrixXd & input_values, Eigen::MatrixXd & results);
    void feed_forward_and_remember(const Eigen::MatrixXd & input_values, std::vector<Eigen::MatrixXd>& each_layers_outputs_);

    void add(const NeuralLayer &layer);
    void insert(uint16_t i, const NeuralLayer &layer);
    void remove(uint16_t i);
    void pop_back();
    void train(const Eigen::MatrixXd & input_values, 
                                        const Eigen::MatrixXd & target_result, 
                                        double learning_scale, double fval);
private:
    void back_propogation__(const Eigen::RowVectorXd & input_values, 
                                        const Eigen::RowVectorXd & target_result, 
                                        double learning_scale, double fval);                                   
private:
    std::vector<NeuralLayer> __layers;
    // Parameters __params;
};
} // fun_learning
#endif //__NEURAL_NETWORK_HPP__