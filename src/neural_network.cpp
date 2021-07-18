#include "neural_network.hpp"
#include "activation_fuctions.hpp"
#include <random>
#include <math.h> 

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

size_t NeuralNetwork::size() const
{
    return __layers.size();
}

NeuralLayer NeuralNetwork::get_layer(uint16_t key) const
{
    return __layers[key];
}

void NeuralNetwork::train(const MatrixXd & input_values, 
                                        const MatrixXd & target_result, 
                                        double learning_scale, double fval, bool shuffle)
{
    double fval_ = fval;
    uint32_t cycle_num = 0;
    uint32_t correct_res = 0; 
    for(uint32_t cycle_num = 0; cycle_num < 1000000; cycle_num++)
    {
        // std::cout << "cycle_num: " << cycle_num << std::endl ;
        std::vector<uint32_t> indexes;
        for(uint32_t i=0; i < input_values.rows(); i++)
            indexes.push_back(i);
        if(shuffle)
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indexes.begin(), indexes.end(), g);
        }
        for(uint16_t i=0; i < input_values.rows();i++)
        {
            if(back_propogation__(input_values.row(indexes[i]), target_result.row(indexes[i]), learning_scale, fval_) <= fval_)
                correct_res++;
        }
        std::cout << "correct_res: " << correct_res << std::endl ;
        if(correct_res == input_values.rows())
        {
            printf("Training is finished with %d itarations!\n", cycle_num );
            return;
        }
        correct_res = 0;
    }
    printf("Training is finished after reaching maximum of itarations!\n" );
    
}

void NeuralNetwork::update_layer_weights(const Eigen::MatrixXd& old_weights, const Eigen::ArrayXXd& delta, 
                            const Eigen::ArrayXXd& inputs, Eigen::MatrixXd& new_weights, double learning_scale)

{
    Eigen::ArrayXXd rdelta = delta.replicate(old_weights.rows(), 1);
    // std::cout << "rdelta =\n" << rdelta << std::endl;
    Eigen::ArrayXXd rinputs = inputs.transpose().replicate(1, old_weights.cols());
    // std::cout << "rinputs =\n" << rinputs << std::endl;
    new_weights = old_weights.array() - learning_scale*rdelta*rinputs;
}

double NeuralNetwork::back_propogation__(const RowVectorXd & input_values, 
                                        const RowVectorXd & target_result, 
                                        double learning_scale, double fval)
{
    
    vector<MatrixXd> each_layers_out;
    MatrixXd results;
    feed_forward_and_remember(input_values, each_layers_out);
    results = each_layers_out.back();
    auto err = 0.5*sqrt(((target_result - results).array().square()).sum());
    if(err > fval)
    {
        ArrayXXd E, dR, inputs, delta;
        Eigen::MatrixXd weights, new_weights;
        uint16_t layers_num = __layers.size() - 1;
        for(int16_t i=layers_num; i >= 0;i--)
        {
            /// Process output layer
            results = each_layers_out[i+1];
            __layers[i].get_weights(weights);
            if (i == layers_num)
                E = -(target_result-results).array();
            else
            {
                __layers[i + 1].get_weights(new_weights);
                E = (delta.matrix()*new_weights.transpose()).array();
            }
                
        
            dR = sigmoid_derivative(results).array();
        
            inputs = each_layers_out[i].array();
            
            delta = E*dR;

            update_layer_weights(weights, delta, inputs, new_weights, learning_scale);
            __layers[i].set_weights(new_weights);
        }
    }
    return err;
}

NeuralLayer& NeuralNetwork::operator[](uint16_t key) 
{
    return __layers[key];
}

YAML::Node YAML::convert<fun_learning::NeuralNetwork>::encode(const fun_learning::NeuralNetwork& rhs)
{
    Node node;
    string name;
    for(size_t i=0; i < rhs.size(); i++)
    {
        node.push_back(rhs.get_layer(i));
    }
    return node;
}

bool YAML::convert<fun_learning::NeuralNetwork>::decode(const YAML::Node& node, fun_learning::NeuralNetwork& rhs)
{
    if(node.IsSequence())
    {
        size_t n = node.size();
        for(uint16_t i=0; i < n; i++)
        {
            NeuralLayer layer = node[i].as<NeuralLayer>();
            rhs.add(layer);
        }
        return true;
    }
    else
        return false;
}

