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
    NeuralNetwork();

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
    NeuralLayer& operator[](uint16_t key);
private:
    double back_propogation__(const Eigen::RowVectorXd & input_values, 
                                        const Eigen::RowVectorXd & target_result, 
                                        double learning_scale, double fval);            
    void update_layer_weights(const Eigen::MatrixXd& old_weights, const Eigen::ArrayXXd& delta, 
                              const Eigen::ArrayXXd& inputs, Eigen::MatrixXd& new_weights, double learning_scale);
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