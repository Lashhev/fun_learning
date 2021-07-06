
#include "neural_network.hpp"
#include <functional>
#include <yaml-cpp/yaml.h>
#include <fstream>

using namespace fun_learning;


int main(int argc, char *argv[])
{
    NeuralNetwork net;
    NeuralLayer layer1(4, 4);
    net.add(layer1);
    NeuralLayer layer2(4, 8);
    net.add(layer2);
    NeuralLayer layer3(8, 5);
    net.add(layer3);

    // Training set 
    Eigen::MatrixXd train_input = Eigen::MatrixXd(15, 4);
    train_input << 0, 0, 0, 0,
                 0, 0, 0, 1, 
                 0, 0, 1, 0, 
                 0, 1, 0, 0, 
                 1, 0, 0, 0, 

                 0, 0, 1, 1, 
                 0, 1, 0, 1, 
                 1, 0, 0, 1, 
                 0, 1, 1, 0, 
                 1, 0, 1, 0, 
                 1, 1, 0, 0, 

                 0, 1, 1, 1,
                 1, 0, 1, 1,
                 1, 1, 0, 1,
                 1, 1, 1, 0;

    Eigen::MatrixXd train_output = Eigen::MatrixXd(15, 5);  
    train_output <<  1, 0, 0, 0, 0,
                    0, 1, 0, 0, 0,
                    0, 1, 0, 0, 0, 
                    0, 1, 0, 0, 0, 
                    0, 1, 0, 0, 0, 

                    0, 0, 1, 0, 0, 
                    0, 0, 1, 0, 0, 
                    0, 0, 1, 0, 0, 
                    0, 0, 1, 0, 0, 
                    0, 0, 1, 0, 0, 
                    0, 0, 1, 0, 0, 

                    0, 0, 0, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 1, 0;


    net.train3(train_input, train_output, 20, 1e-3);
    Eigen::RowVector4d test_input;
    Eigen::MatrixXd test_out;
    test_input << 0, 0, 0, 1;
    net.feed_forward(test_input, test_out);
    std::cout <<test_out << std::endl;
    std::ofstream file1;
    file1.open("/home/lashchev/Documents/MyExperiments/neural_training/libs/fun_learning/data/network.yaml");
    YAML::Node node;
    node = net;
    file1 << node;
    
    
    // YAML::Node node;
    // node = YAML::LoadFile("/home/lashchev/Documents/MyExperiments/neural_training/libs/fun_learning/data/network.yaml");
    // NeuralNetwork net = node.as<NeuralNetwork>();
    // Eigen::RowVector4d test_input;
    // Eigen::MatrixXd test_out;
    // test_input << 1, 0, 0, 0;
    // net.feed_forward(test_input, test_out);
    // std::cout <<test_out << std::endl;

    return 0;
}