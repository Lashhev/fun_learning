
#include "neural_layer.hpp"

using namespace Eigen;
using namespace std;
using namespace fun_learning;

int main(int argc, char* argv[])
{
    MatrixXd x = MatrixXd(3, 4);
    x << 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1;

    // MatrixXd x = MatrixXd(1, 3);
    // x << 1, 0, 1;
    printf("\nx=\n");
    std::cout << x << std::endl;
    // VectorXd weights(3,1);
    // weights << -6, 12, 4;
    // auto p = Perceptron(3, 0.0, weights);
    // auto r = (x*p);
    auto layer = Layer(6,4);
    VectorXd weights(3,1);
    weights << -6, 12, 4;
    MatrixXd w(3,3);
    w.col(0) = weights/2;
    w.col(1) = weights;
    w.col(2) = weights*2;
    // layer.set_weights(w);
    auto y = x*layer;
    for(uint16_t i=0; i < y.cols();i++)
    {
        printf("\ny(%d)=\n", i+1);
        std::cout << y.col(i);
    }
    std::cout << "\ny = " << std::endl;
    std::cout << y << std::endl;

    return 0;
}