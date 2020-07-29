#include <eigen3/Eigen/Dense>

Eigen::VectorXd sigmoid(const Eigen::VectorXd & x, double a=1.0);
Eigen::VectorXd tanh_func(const Eigen::VectorXd &  x, double a=1.0);
Eigen::VectorXd binary_step(const Eigen::VectorXd &  x, double threshold=1.0);