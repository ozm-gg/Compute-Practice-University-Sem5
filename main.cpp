#include <iostream>
#include "VectorMath.h"
#include "MatrixMath.h"
#include "OperationCounter.h"


using CountedDouble = vmath::OpCounter<double>;

int main() {

    vmath::Matrix<CountedDouble> B = vmath::Matrix<CountedDouble>::randomMatrix(7);

    vmath::reset_ops();
    auto tmp = vmath::Matrix<CountedDouble>::solve(B, vmath::Vector<CountedDouble>::randomVector(7));
    std::cout << "Gauss ops:" << vmath::get_ops() << std::endl;

    std::cout << "B:" << B << std::endl;
    vmath::reset_ops();
    auto L = (B * B.transpose()).cholesky_decompose();
    std::cout << "cholesky ops:" << vmath::get_ops() << std::endl;
    std::cout << "L:" << L << std::endl;

    std::cout << "B*B^T:" << B * B.transpose() << std::endl;
    std::cout << "L*L^T:" << L * L.transpose() << std::endl;

    std::cout << "norm B*B^T - L*L^T:" << (B * B.transpose() - L * L.transpose()).norm_frobenius() << std::endl;


    int n = 7;
    vmath::Vector<CountedDouble> a = vmath::Vector<CountedDouble>::randomVector(n - 1);
    vmath::Vector<CountedDouble> b = vmath::Vector<CountedDouble>::randomVector(n);
    vmath::Vector<CountedDouble> c = vmath::Vector<CountedDouble>::randomVector(n - 1);

    std::cout<<a<<std::endl;
    std::cout<<b<<std::endl;
    std::cout<<c<<std::endl;

    vmath::Vector<CountedDouble> x_true = vmath::Vector<CountedDouble>::randomVector(n);
    vmath::Vector<CountedDouble> d = vmath::Matrix<CountedDouble>::threeDiagonal(a, b, c) * x_true;

    std::cout << vmath::Matrix<CountedDouble>::threeDiagonal(a, b, c) << std::endl;

    auto x_sol = vmath::Matrix<CountedDouble>::thomas_solve(a, b, c, d);

    std::cout<<"(x_sol - x_true).norm(): "<<(x_sol - x_true).norm()<<std::endl;


    return 0;
}
