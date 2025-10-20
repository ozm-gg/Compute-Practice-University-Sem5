#include <iostream>
#include "VectorMath.h"
#include "MatrixMath.h"
#include "OperationCounter.h"


using CountedDouble = vmath::OpCounter<double>;

int main() {

    size_t n = 7;
    vmath::Matrix<CountedDouble> A = vmath::Matrix<CountedDouble>::random_diagonally_dominant(n, 100);

    vmath::Vector<CountedDouble> x_true = vmath::Vector<CountedDouble>::randomVector(n);
    vmath::Vector<CountedDouble> b = A * x_true;

    std::cout <<"x_true" << x_true << std::endl;

    vmath::reset_ops();
    auto tmp = vmath::Matrix<CountedDouble>::simple_iteration(A, b);
    std::cout << "simple_iteration ops:" << vmath::get_ops() << std::endl;

    std::cout << "norm(x - x_true)" << (tmp - x_true).norm() << std::endl;

    vmath::reset_ops();
    tmp = vmath::Matrix<CountedDouble>::seidel(A, b);
    std::cout << "seidel ops:" << vmath::get_ops() << std::endl;

    std::cout << "norm(x - x_true)" << (tmp - x_true).norm() << std::endl;


    return 0;
}
