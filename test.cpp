#include <gtest/gtest.h>
#include "VectorMath.h"
#include "MatrixMath.h"

using namespace vmath;

// ====================== TESTS FOR VECTOR ======================
TEST(VectorTest, ConstructorAndAccess) {
    Vector<int> v{1, 2, 3};
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);
}

TEST(VectorTest, ArithmeticOps) {
    Vector<int> a{1, 2, 3};
    Vector<int> b{4, 5, 6};
    auto c = a + b;
    EXPECT_EQ(c, (Vector<int>{5, 7, 9}));
    auto d = b - a;
    EXPECT_EQ(d, (Vector<int>{3, 3, 3}));
    auto e = -a;
    EXPECT_EQ(e, (Vector<int>{-1, -2, -3}));
}

TEST(VectorTest, ScalarOps) {
    Vector<double> v{1.0, -2.0, 3.0};
    EXPECT_EQ(v * 2.0, (Vector<double>{2.0, -4.0, 6.0}));
    EXPECT_EQ(2.0 * v, (Vector<double>{2.0, -4.0, 6.0}));
    EXPECT_EQ(v / 2.0, (Vector<double>{0.5, -1.0, 1.5}));
}

TEST(VectorTest, DotAndNorms) {
    Vector<double> a{1.0, 2.0, 3.0};
    Vector<double> b{4.0, -5.0, 6.0};
    EXPECT_DOUBLE_EQ(a.dot(b), 12.0);
    EXPECT_DOUBLE_EQ(a.norm_squared(), 14.0);
    EXPECT_NEAR(a.norm(), std::sqrt(14.0), 1e-12);
    EXPECT_EQ(a.norm1(), 6.0);
    EXPECT_EQ(b.norm_inf(), 6.0);
}

TEST(VectorTest, NormalizationAndDistance) {
    Vector<double> a{3.0, 4.0};
    auto an = a.normalized();
    EXPECT_NEAR(an.norm(), 1.0, 1e-12);
    Vector<double> b{0.0, 0.0};
    EXPECT_THROW(b.normalized(), std::domain_error);

    Vector<double> c{1.0, 2.0};
    Vector<double> d{4.0, 6.0};
    EXPECT_NEAR(c.distance_to(d), 5.0, 1e-12);
}

TEST(VectorTest, ProjectionAndCross) {
    Vector<double> a{1.0, 2.0, 3.0};
    Vector<double> b{0.0, 1.0, 0.0};
    auto p = a.projection_onto(b);
    EXPECT_EQ(p, (Vector<double>{0.0, 2.0, 0.0}));

    Vector<double> x{1, 0, 0}, y{0, 1, 0};
    auto z = x.cross(y);
    EXPECT_EQ(z, (Vector<double>{0, 0, 1}));
    EXPECT_THROW(x.cross(Vector<double>{1, 2}), std::domain_error);
}

// ====================== TESTS FOR MATRIX ======================
TEST(MatrixTest, ConstructorAndAccess) {
    Matrix<int> m{{1, 2}, {3, 4}};
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 2);
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(1, 1), 4);
}

TEST(MatrixTest, IdentityAndZeros) {
    auto I = Matrix<int>::identity(3);
    EXPECT_EQ(I(0,0), 1);
    EXPECT_EQ(I(1,1), 1);
    EXPECT_EQ(I(2,2), 1);
    EXPECT_EQ(I(0,1), 0);

    auto Z = Matrix<int>::zeros(2,3);
    EXPECT_EQ(Z(1,2), 0);
}

TEST(MatrixTest, ArithmeticOps) {
    Matrix<int> A{{1,2},{3,4}};
    Matrix<int> B{{5,6},{7,8}};
    auto C = A + B;
    EXPECT_EQ(C(0,0), 6);
    EXPECT_EQ(C(1,1), 12);
    auto D = B - A;
    EXPECT_EQ(D(0,0), 4);
    auto E = -A;
    EXPECT_EQ(E(0,0), -1);
}

TEST(MatrixTest, ScalarOps) {
    Matrix<int> A{{1,2},{3,4}};
    auto B = A * 2;
    EXPECT_EQ(B(0,1), 4);
    auto C = B / 2;
    EXPECT_EQ(C(1,0), 3);
}

TEST(MatrixTest, MatrixMultiplication) {
    Matrix<int> A{{1,2,3},{4,5,6}};
    Matrix<int> B{{7,8},{9,10},{11,12}};
    auto C = A * B;
    EXPECT_EQ(C.rows(), 2);
    EXPECT_EQ(C.cols(), 2);
    EXPECT_EQ(C(0,0), 58);
    EXPECT_EQ(C(1,1), 154);
}

TEST(MatrixTest, MatrixVectorMultiplication) {
    Matrix<int> A{{1,2},{3,4}};
    Vector<int> v{5,6};
    auto res = A * v;
    EXPECT_EQ(res[0], 17);
    EXPECT_EQ(res[1], 39);
}

TEST(MatrixTest, Transpose) {
    Matrix<int> A{{1,2,3},{4,5,6}};
    auto T = A.transpose();
    EXPECT_EQ(T.rows(), 3);
    EXPECT_EQ(T.cols(), 2);
    EXPECT_EQ(T(1,0), 2);
    EXPECT_EQ(T(2,1), 6);
}

TEST(MatrixTest, Norms) {
    Matrix<int> A{{1,-2},{3,-4}};
    EXPECT_EQ(A.norm1(), 6);
    EXPECT_EQ(A.norm_inf(), 7);
    EXPECT_EQ(A.norm_frobenius(), 5);
}

TEST(MatrixTest, DeterminantAndInverse) {
    Matrix<double> A{{4,7},{2,6}};
    double det = A.determinant_via_lup();
    EXPECT_NEAR(det, 10.0, 1e-12);
    auto inv = A.inverse();
    EXPECT_NEAR(inv(0,0), 0.6, 1e-12);
    EXPECT_NEAR(inv(0,1), -0.7, 1e-12);
    EXPECT_NEAR(inv(1,0), -0.2, 1e-12);
    EXPECT_NEAR(inv(1,1), 0.4, 1e-12);
}

TEST(MatrixTest, SolveSystem) {
    Matrix<double> A{{3,2},{1,2}};
    Vector<double> b{5,5};
    auto x = Matrix<double>::solve(A,b);
    EXPECT_NEAR(x[0], 0.0, 1e-12);
    EXPECT_NEAR(x[1], 2.5, 1e-12);
}

template<typename T>
void expect_vector_near(const Vector<T>& v, const Vector<T>& ref, double tol = 1e-9) {
    ASSERT_EQ(v.size(), ref.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(v[i]), static_cast<double>(ref[i]), tol) << "index " << i;
    }
}

template<typename T>
void expect_matrix_near(const Matrix<T>& A, const Matrix<T>& B, double tol = 1e-9) {
    ASSERT_EQ(A.rows(), B.rows());
    ASSERT_EQ(A.cols(), B.cols());
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            EXPECT_NEAR(static_cast<double>(A(i,j)), static_cast<double>(B(i,j)), tol) << "element ("<<i<<","<<j<<")";
}

// ========== ТЕСТЫ ДЛЯ CHOLESKY ==========
TEST(CholeskyTest, Simple2x2) {
    Matrix<double> A{{4.0, 1.0}, {1.0, 3.0}}; // SPD
    auto L = A.cholesky_decompose();

    // L should be lower triangular with positive diagonal
    EXPECT_NEAR(L(0,0), 2.0, 1e-12);
    EXPECT_NEAR(L(1,0), 0.5, 1e-12);
    EXPECT_NEAR(L(1,1), std::sqrt(3.0 - 0.25), 1e-12); // sqrt(11/4) = sqrt(2.75)

    // Reconstruction
    auto recon = L * L.transpose();
    expect_matrix_near(recon, A, 1e-9);
}

TEST(CholeskyTest, Classic3x3) {
    // symmetric positive definite (classic test)
    Matrix<double> A{
        {25.0, 15.0, -5.0},
        {15.0, 18.0,  0.0},
        {-5.0,  0.0, 11.0}
    };
    auto L = A.cholesky_decompose();
    // check lower-triangular structure and diag positive
    EXPECT_NEAR(L(0,0), 5.0, 1e-12);
    EXPECT_NEAR(L(1,0), 3.0, 1e-12);
    EXPECT_NEAR(L(2,0), -1.0, 1e-12);
    EXPECT_GT(L(0,0), 0.0);
    EXPECT_GT(L(1,1), 0.0);
    EXPECT_GT(L(2,2), 0.0);

    auto recon = L * L.transpose();
    expect_matrix_near(recon, A, 1e-9);
}

TEST(CholeskyTest, NonPositiveDefiniteThrows) {
    // symmetric but not positive definite
    Matrix<double> A{{1.0, 2.0}, {2.0, 1.0}};
    EXPECT_THROW(A.cholesky_decompose(), std::runtime_error);
}

TEST(CholeskyTest, OneByOne) {
    Matrix<double> A{{4.0}};
    auto L = A.cholesky_decompose();
    EXPECT_EQ(L.rows(), 1u);
    EXPECT_EQ(L.cols(), 1u);
    EXPECT_NEAR(L(0,0), 2.0, 1e-12);
}

// ========== ТЕСТЫ ДЛЯ THOMAS (трёхдиагоналка) ==========
TEST(ThomasTest, Small3x3ExactSolution) {
    Vector<double> a{1.0, 1.0};      // sub-diagonal (n-1)
    Vector<double> b{2.0, 2.0, 2.0}; // main diagonal (n)
    Vector<double> c{3.0, 3.0};      // super-diagonal (n-1)

    // choose x_true and compute d = A * x_true
    Vector<double> x_true{1.0, 2.0, 3.0};
    // compute d manually
    Vector<double> d(3);
    d[0] = b[0]*x_true[0] + c[0]*x_true[1];
    d[1] = a[0]*x_true[0] + b[1]*x_true[1] + c[1]*x_true[2];
    d[2] = a[1]*x_true[1] + b[2]*x_true[2];

    auto x_sol = Matrix<double>::thomas_solve(a, b, c, d);
    expect_vector_near(x_sol, x_true, 1e-12);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
