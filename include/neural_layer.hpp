#ifndef __NEURON_LAYER_HPP__
#define __NEURON_LAYER_HPP__

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <functional>
#include "version_info.h"
#include <yaml-cpp/yaml.h>

namespace fun_learning
{
template<typename MatrixT>
void printM(const MatrixT& M, const std::string& name)
{
    std::cout << name << " = \n" << M << std::endl << std::endl;
}
class NeuralLayer
{
public:
    NeuralLayer();
    NeuralLayer(uint16_t input_number, uint16_t neuron_number, const Eigen::VectorXd &bias, const Eigen::MatrixXd& weights, std::string activation_func="sigmoid");
    NeuralLayer(uint16_t input_number,uint16_t neuron_number=1,  double min_val=-0.1, double max_val=0.1, std::string activation_func="sigmoid");
    
    void set_bias(const Eigen::VectorXd &bias);

    void set_weights(const Eigen::MatrixXd& weights);

    void set_activation_func(const std::string &func_type);

    void get_weights(Eigen::MatrixXd& weights) const;

    void get_biases(Eigen::VectorXd &bias) const;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd, double)>& get_active_func();

    void add_neuron(const Eigen::VectorXd &weights, double bias);
    void remove(uint16_t n);
    void clear();

protected:
// EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void __choose_activation_func(const std::string& func_type);

    void __set_random_weights(double min_val=-5, double max_val=5);
    
    Eigen::MatrixXd __synaptic_weights;
    uint16_t __weights_number;
    uint16_t __neurons_number;
    Eigen::VectorXd __bias;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd, double)> __activation_func;
};

Eigen::MatrixXd operator*(const Eigen::MatrixXd& input_values, NeuralLayer p);
} //fun_learning
namespace YAML
{
template <>
struct convert<fun_learning::NeuralLayer> {
  static Node encode(const fun_learning::NeuralLayer& rhs);

  YAML_CPP_API static bool decode(const Node& node, fun_learning::NeuralLayer& rhs);
};
} // YAML
#endif // __NEURON_LAYER_HPP__