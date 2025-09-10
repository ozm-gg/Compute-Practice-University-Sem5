#include <iostream>
#include "VectorMath.h"
#include "MatrixMath.h"


int main() {
    vmath::Matrix<double> A{{2.0, 3.0, 1.0}, {4.0, 7.0, 7.0}, {6.0, 18.0, 22.0}};

    std::cout << "A:" << A << std::endl;


    auto [L, U, P] = A.lup_decompose();
    std::cout << "L:" << L << std::endl;
    std::cout << "U:" << U << std::endl;
    std::cout << "P: "<< P << std::endl;

    vmath::Matrix<double> B = vmath::Matrix<double>::identity(3);

    std::cout << "A:\n" << A << "\n";
    std::cout << "A + I:\n" << (A + B) << "\n";
    std::cout << "A - I:\n" << (A - B) << "\n";
    std::cout << "2*A:\n" << (2.0 * A) << "\n";

    vmath::Vector<double> x_true{1.0, 2.0, 3.0};
    auto x = vmath::Matrix<double>::solve(A, A * x_true);

    std::cout << "Solution x:"<< x << std::endl;
    std::cout << "check: "<< A * x << std::endl;
    std::cout << "Err: "<< (x - x_true).norm() << std::endl;


    std::cout << "det(A) = " << A.determinant_via_lup() << "\n";

    std::cout << "A^{-1}:\n" << A.inverse() << "\n";
    return 0;

    return 0;
}

