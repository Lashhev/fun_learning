
#include "neural_network.hpp"

using namespace Eigen;
using namespace std;
using namespace fun_learning;

// int main(int argc, char* argv[])
// {
//     RowVectorXd x = RowVectorXd(3);
//     x << 1, 0, 1;//, 1, 1, 0;//, 0, 1, 1, 0, 1, 1;
//     RowVectorXd x_t = RowVectorXd(3);
//     x_t << 0, 1, 0;//, 1, 1, 0;//, 0, 1, 1, 0, 1, 1;

//     // MatrixXd x = MatrixXd(1, 3);
//     // x << 1, 0, 1;
//     printf("\nx=\n");
//     std::cout << x << std::endl;
//     // VectorXd weights(3,1);
//     // weights << -6, 12, 4;
//     // auto p = Perceptron(3, 0.0, weights);
//     // auto r = (x*p);
//     auto layer = NeuralLayer(3,3,-0.1,0.1);
//     VectorXd weights(3,1);
//     VectorXd bias(3,1);
//     bias << -2, 1, 2;
//     weights << -6, 12, 4;
//     MatrixXd w(3,3);
//     w.col(0) = weights/2;
//     w.col(1) = weights;
//     w.col(2) = weights*2;
//     // w.col(3) = weights*2;
//     // layer.set_weights(w);
//     // layer.set_bias(bias);
//     // auto y = x*layer;
//     layer.back_propogation(x, x_t, 80, 1e-12);
//     // for(uint16_t i=0; i < y.cols();i++)
//     // {
//     //     printf("\ny(%d)=\n", i+1);
//     //     std::cout << y.col(i);
//     // }
//     auto y = x*layer;
//     std::cout << "\ny = " << std::endl;
//     std::cout << y << std::endl;

//     return 0;
// }

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
    network1.add(layer1);
    network1.add(layer2);
    // MatrixXd y;
    // network1.feed_forward(x, y);
    // std::cout << "yh = \n" << y << std::endl << std::endl;
    // network1.add(layer2);
    // MatrixXd y2;
    // network1.feed_forward(x, y2);
    // std::cout << "yo = \n" << y2 << std::endl << std::endl;
    // MatrixXd result;
    network1.train(x, t_y, 90, 1e-6);
    MatrixXd y;
    network1.feed_forward(x, y);
    printM(y, "y");

    // network1.fast_forward(x, result);
    // std::cout << "\nresult = " << std::endl;
    // std::cout << result << std::endl;
    return 0;
}