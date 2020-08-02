#include <eigen3/Eigen/Dense>
namespace fun_learning
{

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd & x, double a=1.0);
Eigen::MatrixXd tanh_func(const Eigen::MatrixXd &  x, double a=1.0);
Eigen::MatrixXd binary_step(const Eigen::MatrixXd &  x, double threshold=1.0);
Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd &  x);

} //fun_learning