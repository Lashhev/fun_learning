
#include "neural_network.hpp"
#include <fstream>

using namespace Eigen;
using namespace std;
using namespace fun_learning;

int main(int argc, char* argv[])
{
    MatrixXd Wh(3, 2), Wo(2, 3);
    Wh << -0.1, 0.2, 0.1, -0.05, 0.11, 0.15;
    Wo << 0.12, -1.05, 0.05, -0.04, 0.08, 0.15;
    RowVectorXd x(3);
    x << 1,0,1;
    RowVectorXd t_y(3);
    t_y << 1, 0, 0;
    NeuralNetwork network1 = NeuralNetwork();
    auto layer1 = NeuralLayer(3,2,-0.1,0.1);
    layer1.set_weights(Wh);
    auto layer2 = NeuralLayer(2,3,-0.1,0.1);
    layer2.set_weights(Wo);
    YAML::Node node1;
    node1 = YAML::LoadFile("/home/lashchev/Documents/fun_learning/data/neurons.yaml");
    layer1 = node1.as<NeuralLayer>();
    network1.add(layer1);
    network1.add(layer2);
    network1.train(x, t_y, 90, 1e-6);
    MatrixXd y;
    network1.feed_forward(x, y);
    printM(y, "y");
    YAML::Node node2;
    ofstream file;
    file.open("/home/lashchev/Documents/fun_learning/data/neurons.yaml");
    node2 = network1[0];
    file << node2;
    file.close();
    return 0;
}