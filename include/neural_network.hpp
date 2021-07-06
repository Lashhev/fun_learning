#ifndef __NEURAL_NETWORK_HPP__
#define __NEURAL_NETWORK_HPP__

#include "neural_layer.hpp"
#include <vector>
#include <yaml-cpp/yaml.h>
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

    NeuralLayer get_layer(uint16_t key) const;
    void add(const NeuralLayer &layer);
    void insert(uint16_t i, const NeuralLayer &layer);
    void remove(uint16_t i);
    void pop_back();
    size_t size() const;
    void train(const Eigen::MatrixXd & input_values, 
                                        const Eigen::MatrixXd & target_result, 
                                        double learning_scale, double fval);
    void train2(const Eigen::MatrixXd & input_values, 
                                        const Eigen::MatrixXd & target_result, 
                                        double learning_scale, double fval);
    void train3(const Eigen::MatrixXd & input_values, 
                                        const Eigen::MatrixXd & target_result, 
                                        double learning_scale, double fval);
    NeuralLayer& operator[](uint16_t key);
private:
    double back_propogation__(const Eigen::RowVectorXd & input_values, 
                                        const Eigen::RowVectorXd & target_result, 
                                        double learning_scale, double &fval);  
    void back_propogation2__(const Eigen::RowVectorXd & input_values, 
                                        const Eigen::RowVectorXd & target_result, 
                                        double learning_scale, double &fval);  
                                     
private:
    std::vector<NeuralLayer> __layers;
    // Parameters __params;
};
} // fun_learning

namespace YAML
{
template <>
struct convert<fun_learning::NeuralNetwork> {
  static Node encode(const fun_learning::NeuralNetwork& rhs);

  YAML_CPP_API static bool decode(const Node& node, fun_learning::NeuralNetwork& rhs);
};
} // YAML
#endif //__NEURAL_NETWORK_HPP__