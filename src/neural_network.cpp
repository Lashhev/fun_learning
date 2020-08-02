#include "neural_network.hpp"
#include "activation_fuctions.hpp"

using namespace Eigen;
using namespace std;
using namespace fun_learning;


NeuralNetwork::NeuralNetwork()
{

}

void NeuralNetwork::add(const NeuralLayer &layer)
{
    __layers.push_back(layer);
}

void NeuralNetwork::feed_forward(const MatrixXd & input_values, MatrixXd & results)
{
    results = input_values;
    for(auto layer:__layers)
        results = results*layer;
}

void NeuralNetwork::feed_forward_and_remember(const MatrixXd & input_values, std::vector<Eigen::MatrixXd>& each_layers_outputs_)
{
    MatrixXd results = MatrixXd(input_values);
    each_layers_outputs_.push_back(results);
    for(auto layer:__layers)
    {
        results = results*layer;
        each_layers_outputs_.push_back(results);
    }
}

void NeuralNetwork::insert(uint16_t i, const NeuralLayer &layer)
{
    __layers.insert(__layers.begin()+i, layer);
}

void NeuralNetwork::remove(uint16_t i)
{
    __layers.erase(__layers.begin()+i);
}

void NeuralNetwork::pop_back()
{
    __layers.pop_back();
}

void NeuralNetwork::train(const MatrixXd & input_values, 
                                        const MatrixXd & target_result, 
                                        double learning_scale, double fval)
{
    for(uint16_t i=0; i < input_values.rows();i++)
    {
        back_propogation__(input_values.row(i), target_result.row(i), learning_scale, fval);
    }
    std::cout << "Training is finished! " << std::endl;
}


void NeuralNetwork::back_propogation__(const RowVectorXd & input_values, 
                                        const RowVectorXd & target_result, 
                                        double learning_scale, double fval)
{
    uint32_t cycle_num = 0;
    vector<MatrixXd> each_layers_out;
    MatrixXd results;
    feed_forward_and_remember(input_values, each_layers_out);
    results = each_layers_out.back();
    double e_t = (0.5*(target_result - results).array().square()).sum();
    while (e_t > fval)
    {
        cycle_num++;
        feed_forward_and_remember(input_values, each_layers_out);
        results = each_layers_out.back();
        e_t = (0.5*(target_result - results).array().square()).sum();
        std::cout << "cycle_num: " << cycle_num << std::endl << "Neuron precision: " << e_t << std::endl;
        
        ArrayXXd E, dR, inputs, delta;
        Eigen::MatrixXd weights, new_weights;
        uint16_t layers_num = __layers.size() - 1;
        for(uint16_t i=layers_num; i > 0;i--)
        {
            /// Process output layer
            results = each_layers_out[i+1];
            __layers[i].get_weights(weights);
            if (i == layers_num)
                E = -(target_result-results).array();
            else
                E = delta.matrix()*weights;
        
            dR = sigmoid_derivative(results).array();
       
            inputs = each_layers_out[i].array();
            
            delta = ArrayXXd(E*dR);
        
            new_weights = weights.array() - learning_scale*delta.replicate(weights.rows(), 1)*inputs.array().transpose().replicate(1, weights.cols());
            __layers.back().set_weights(new_weights);
        }
    } 
}

NeuralLayer& NeuralNetwork::operator[](uint16_t key)
{
    return __layers[key];
}

// void NeuralNetwork::back_propogation(const Eigen::RowVectorXd & input_values, 
//                                         Eigen::RowVectorXd & target_result, 
//                                         double learning_scale, double fval)
// {
//     double e_t = fval+50;
//     uint32_t cycle_num = 0;
//     vector<MatrixXd> each_layers_out;
//     MatrixXd results;
//     while (e_t > fval)
//     {
//         cycle_num++;
//         feed_forward_and_remember(input_values, each_layers_out);
//         results = each_layers_out.back();
//         e_t = (0.5*(target_result - results).array().square()).sum();
//         std::cout << "cycle_num: " << cycle_num << std::endl << "Neuron precision: " << e_t << std::endl;
//         // std::cout << "cicle_num: " << cycle_num << std::endl << "Neuron precision: " << e_t << std::endl;

//         /// Process output layer
//         ArrayXXd E = -(target_result-results).array();
//         // printM(E, "E");
//         ArrayXXd dR = sigmoid_derivative(results).array();
//         // printM(dR, "dR");
//         ArrayXXd inputs = (*(each_layers_out.end()-2)).array();
//         // printM(inputs, "inputs");
//         ArrayXXd delta = ArrayXXd(E*dR);
//         // printM(delta, "delta");
//         MatrixXd weights;
//         __layers.back().get_weights(weights);
//         Eigen::MatrixXd new_weights = weights.array() - learning_scale*delta.replicate(weights.rows(), 1)*inputs.array().transpose().replicate(1, weights.cols());
//         __layers.back().set_weights(new_weights);
//         // printM(new_weights, "new_weights");

//         /// Process hidden layer
//         results = *(each_layers_out.end()-2);
//         (*(__layers.end()-2)).get_weights(weights);
//         E = delta.matrix()*weights;
//         dR = sigmoid_derivative(results).array();
//         inputs = (*(each_layers_out.end()-3)).array();
//         delta = ArrayXXd(E*dR);
//         new_weights = weights.array() - learning_scale*delta.replicate(weights.rows(), 1)*inputs.array().transpose().replicate(1, weights.cols());
//         (*(__layers.end()-2)).set_weights(new_weights);

//     } 
//     std::cout << "Training is finished! " << std::endl;
//     feed_forward(input_values, results);
//     printM(results, "results");
// }
