#ifndef __PERCEPTRON_HPP__
#define __PERCEPTRON_HPP__

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <random>
#include <version_info.h>

namespace fun_learning
{
class Perceptron
{
public:
    Perceptron(uint16_t input_number, double min_val=-5, double max_val=5);
    Perceptron(uint16_t input_number, double bias, const Eigen::VectorXd& weights, std::string activation_func);
    void set_bias(double bias);

    void set_weights(const Eigen::VectorXd& weights);

    void set_activation_func(const std::string &func_type);

    void get_weights(Eigen::VectorXd& weights) const;

    void get_bias(double& bias) const;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd, double)>& get_active_func();
protected:
// EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void __choose_activation_func(const std::string& func_type);

    void __set_random_weights(uint16_t weight_number, double min_val=-5, double max_val=5);
    
    Eigen::VectorXd __synaptic_weights;
    uint16_t __weights_number;
    double __bias;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd, double)> __activation_func;
};

Eigen::VectorXd operator*(const Eigen::MatrixXd& input_values, Perceptron p);
} //fun_learning
#endif // __PERCEPTRON_HPP_