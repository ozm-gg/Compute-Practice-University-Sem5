//
// Created by dmits on 08.09.2025.
//

#ifndef MATRIXMATH_H
#define MATRIXMATH_H

#include "VectorMath.h"

#include <vector>
#include <cstddef>
#include <initializer_list>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <type_traits>

#include "OperationCounter.h"

namespace vmath {
    template<typename T>
    class Matrix {
        //static_assert(std::is_arithmetic<T>::value, "Matrix requires arithmetic type");

    public:
        // Конструкторы
        Matrix() = default;

        Matrix(const std::size_t rows, const std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {
        }

        explicit Matrix(const std::size_t rows) : rows_(rows), cols_(rows), data_(rows * rows) {
        }

        Matrix(const std::size_t rows, const std::size_t cols, const T &value) : rows_(rows), cols_(cols),
            data_(rows * cols, value) {
        }

        Matrix(std::initializer_list<std::initializer_list<T> > init) {
            rows_ = init.size();
            cols_ = init.begin()->size();
            data_.reserve(rows_ * cols_);
            for (auto &r: init) {
                if (r.size() != cols_) throw std::invalid_argument("All rows must have the same number of columns");
                for (auto &v: r) data_.push_back(v);
            }
        }

        // Размер и доступ
        std::size_t rows() const noexcept { return rows_; }
        std::size_t cols() const noexcept { return cols_; }
        bool empty() const noexcept { return data_.empty(); }

        // доступ в стиле (i,j)
        T &operator()(const std::size_t i, const std::size_t j) {
            return data_.at(i * cols_ + j);
        }

        const T &operator()(const std::size_t i, const std::size_t j) const {
            return data_.at(i * cols_ + j);
        }

        // небезопасный прямой доступ без проверки
        T &at_unchecked(const std::size_t i, const std::size_t j) noexcept { return data_[i * cols_ + j]; }
        const T &at_unchecked(const std::size_t i, const std::size_t j) const noexcept { return data_[i * cols_ + j]; }

        vmath::Vector<T> row(std::size_t i) const {
            if (i >= rows_) throw std::out_of_range("row out of range");
            vmath::Vector<T> r(cols_);
            for (std::size_t j = 0; j < cols_; ++j) r[j] = (*this)(i, j);
            return r;
        }

        vmath::Vector<T> col(std::size_t j) const {
            if (j >= cols_) throw std::out_of_range("col out of range");
            vmath::Vector<T> c(rows_);
            for (std::size_t i = 0; i < rows_; ++i) c[i] = (*this)(i, j);
            return c;
        }

        Matrix cut(const std::size_t start_row, const std::size_t end_row, const std::size_t start_col,
                   const std::size_t end_col) {
            if (start_row > end_row || start_col > end_col || end_row > rows_ || end_col > cols_) throw
                    std::invalid_argument("Wrong Indexes");
            Matrix res(end_row - start_row, end_col - start_col);
            for (std::size_t i = start_row; i < end_row; ++i) {
                for (std::size_t j = start_col; j < end_col; ++j) {
                    res(i - start_row, j - start_col) = (*this)(i, j);
                }
            }

            return res;
        }

        // Статические фабрики
        static Matrix identity(std::size_t n) {
            Matrix I(n, n, T(0));
            for (std::size_t i = 0; i < n; ++i) I.at_unchecked(i, i) = T(1);
            return I;
        }

        static Matrix randomMatrix(std::size_t n) {
            Matrix m(n, n);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    m(i, j) = static_cast<T>(((float) rand() / (float) rand()));
                }
            }
            return m;
        }

        static Matrix threeDiagonal(const vmath::Vector<T>& a,
                                     const vmath::Vector<T>& b,
                                     const vmath::Vector<T>& c) {
            std::size_t n = b.size();
            Matrix m(n, n);
            m(0,0) = b[0];
            m(1, 0) = a[0];
            for (std::size_t i = 1; i < n - 1; ++i) {
                m(i, i) = b[i];
                m(i + 1, i) = a[i];
                m(i - 1, i) = c[i - 1];
            }
            m(n - 1,n - 1) = b[n - 1];
            m(n - 2, n - 1) = c[n - 2];
            return m;
        }

        static Matrix gilbert(std::size_t n) {
            vmath::Matrix<long double> m(n);

            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    m(i, j) = 1.0 / (static_cast<long double>(1 + i + j));
                }
            }
            return m;
        }

        static Matrix zeros(std::size_t r, std::size_t c) { return Matrix(r, c, T(0)); }
        static Matrix ones(std::size_t r, std::size_t c) { return Matrix(r, c, T(1)); }

        static Matrix concatenate_right(const Matrix &a, const Matrix &b) {
            if (a.rows_ != b.rows_) throw std::invalid_argument("Matrices must match for concatenation");
            Matrix res(a.rows_, b.cols_ + a.cols_);
            for (std::size_t i = 0; i < a.rows_; ++i) {
                for (std::size_t j = 0; j < a.cols_; ++j) {
                    res.at_unchecked(i, j) = a.at_unchecked(i, j);
                }
                for (std::size_t j = 0; j < b.cols_; ++j) {
                    res.at_unchecked(i, j + a.cols_) = b.at_unchecked(i, j);
                }
            }
            return res;
        }

        // Операции
        Matrix transpose() const {
            Matrix t(cols_, rows_);
            for (std::size_t i = 0; i < rows_; ++i)
                for (std::size_t j = 0; j < cols_; ++j)
                    t.at_unchecked(j, i) = (*this)(i, j);
            return t;
        }


        Matrix &operator+=(const Matrix &other) {
            if (rows_ != other.rows_ || cols_ != other.cols_) throw std::invalid_argument(
                "Matrix sizes must match for addition");
            for (std::size_t i = 0; i < data_.size(); ++i) data_[i] += other.data_[i];
            return *this;
        }


        Matrix &operator-=(const Matrix &other) {
            if (rows_ != other.rows_ || cols_ != other.cols_) throw std::invalid_argument(
                "Matrix sizes must match for subtraction");
            for (std::size_t i = 0; i < data_.size(); ++i) data_[i] -= other.data_[i];
            return *this;
        }

        Matrix operator-() const {
            Matrix result = Matrix(rows_, cols_, T(0)) - *this;
            return result;
        }


        Matrix &operator*=(const T &scalar) {
            for (auto &x: data_) x *= scalar;
            return *this;
        }

        Matrix &operator/=(const T &scalar) {
            for (auto &x: data_) x /= scalar;
            return *this;
        }

        friend Matrix operator+(Matrix a, const Matrix &b) {
            a += b;
            return a;
        }

        friend Matrix operator-(Matrix a, const Matrix &b) {
            a -= b;
            return a;
        }

        friend Matrix operator*(Matrix a, const T &scalar) {
            a *= scalar;
            return a;
        }

        friend Matrix operator*(const T &scalar, Matrix a) {
            a *= scalar;
            return a;
        }

        friend Matrix operator/(Matrix a, const T &scalar) {
            a /= scalar;
            return a;
        }

        Matrix operator*(const Matrix &other) const {
            if (cols_ != other.rows_) throw std::invalid_argument("Incompatible sizes for multiplication");
            Matrix res(rows_, other.cols_, T(0));
            for (std::size_t i = 0; i < rows_; ++i) {
                for (std::size_t k = 0; k < cols_; ++k) {
                    T aik = (*this)(i, k);
                    for (std::size_t j = 0; j < other.cols_; ++j) {
                        res.at_unchecked(i, j) += aik * other.at_unchecked(k, j);
                    }
                }
            }
            return res;
        }

        vmath::Vector<T> operator*(const vmath::Vector<T> &v) const {
            if (cols_ != v.size()) throw std::invalid_argument("Incompatible sizes for matrix-vector multiplication");
            vmath::Vector<T> res(rows_, T(0));
            for (std::size_t i = 0; i < rows_; ++i) {
                long double sum = 0;
                for (std::size_t j = 0; j < cols_; ++j) sum += (T) (*this)(i, j) * v[j];
                res[i] = (T) sum;
            }
            return res;
        }

        void swap_rows(size_t i, size_t j) {
            for (std::size_t k = 0; k < cols_; ++k) {
                std::swap((*this)(i, k), (*this)(j, k));
            }
        }

        // нормы
        T norm1() const {
            T max_sum = T(0);
            for (std::size_t j = 0; j < cols_; j++) {
                T col_sum = T(0);
                for (std::size_t i = 0; i < rows_; i++)
                    col_sum += std::abs((*this)(i, j));
                max_sum = std::max(max_sum, col_sum);
            }
            return max_sum;
        }

        T norm_inf() const {
            T max_sum = T(0);
            for (std::size_t i = 0; i < rows_; i++) {
                T row_sum = T(0);
                for (std::size_t j = 0; j < cols_; j++)
                    row_sum += std::abs((*this)(i, j));
                max_sum = std::max(max_sum, row_sum);
            }
            return max_sum;
        }

        T norm_frobenius() const {
            T sum = T(0);
            for (std::size_t i = 0; i < rows_; i++) {
                for (std::size_t j = 0; j < cols_; j++) {
                    sum += (*this)(i, j) * (*this)(i, j);
                }
            }
            return std::sqrt(sum);
        }


        // Числа обусловленности
        T condition_number_1() const { return this->norm1() * this->inverse().norm1(); }

        T condition_number_inf() const { return this->norm_inf() * this->inverse().norm_inf(); }

        T condition_number_frobenius() const { return this->norm_frobenius() * this->inverse().norm_frobenius(); }


        // LUP разложение
        std::tuple<Matrix, Matrix, Matrix> lup_decompose(
            long double tol = std::numeric_limits<long double>::epsilon()) const {
            if (rows_ != cols_) throw std::invalid_argument("LUP decomposition requires a square matrix");
            std::size_t n = rows_;
            Matrix U = *this;
            Matrix L = Matrix<T>::identity(n);
            Matrix P = Matrix<T>::identity(n);

            for (std::size_t i = 0; i < n; i++) {
                T pivotValue = T(0);
                std::size_t pivot = i;
                for (std::size_t row = i; row < n; row++) {
                    if (std::abs(U(row, i)) > pivotValue) {
                        pivotValue = std::abs(U(row, i));
                        pivot = row;
                    }
                }

                if (pivotValue <= (T) tol) {
                    throw std::runtime_error("Matrix is singular to working precision");
                }

                if (pivot != i) {
                    U.swap_rows(pivot, i);
                    P.swap_rows(pivot, i);
                    if (i > 0) {
                        for (std::size_t k = 0; k < i; k++) {
                            std::swap(L(i, k), L(pivot, k));
                        }
                    }
                }
                for (std::size_t j = i + 1; j < n; j++) {
                    L(j, i) = U(j, i) / U(i, i);
                    for (std::size_t k = i; k < n; k++) {
                        U(j, k) -= L(j, i) * U(i, k);
                    }
                }
            }

            return {L, U, P};
        }


        //Холецкий
        Matrix cholesky_decompose(long double tol = std::numeric_limits<long double>::epsilon()) const {
            if (rows_ != cols_)
                throw std::invalid_argument("Cholesky decomposition requires a square matrix");

            std::size_t n = rows_;
            Matrix L(n, n, T(0));

            for (std::size_t i = 0; i < n; i++) {
                for (std::size_t j = 0; j <= i; j++) {
                    T sum = (*this)(i, j);
                    for (std::size_t k = 0; k < j; k++) {
                        sum -= (T) L(i, k) * L(j, k);
                    }

                    if (i == j) {
                        if (sum <= (T) tol)
                            throw std::runtime_error("Matrix is not positive definite");
                        L(i, j) = std::sqrt(sum);
                    } else {
                        L(i, j) = sum / L(j, j);
                    }
                }
            }

            return L;
        }

        //Прогонка
        static vmath::Vector<T> thomas_solve(const vmath::Vector<T>& a,
                                     const vmath::Vector<T>& b,
                                     const vmath::Vector<T>& c,
                                     const vmath::Vector<T>& d) {
            std::size_t n = b.size();
            if (d.size() != n) throw std::invalid_argument("Sizes do not match");
            if (a.size() != n - 1 || c.size() != n - 1)
                throw std::invalid_argument("Sub/super-diagonals have wrong size");

            vmath::Vector<T> cp(n - 1);
            vmath::Vector<T> dp(n);

            cp[0] = c[0] / b[0];
            dp[0] = d[0] / b[0];

            // Прямой ход
            for (std::size_t i = 1; i < n; ++i) {
                T denom = b[i] - a[i - 1] * cp[i - 1];
                if (i < n - 1) cp[i] = c[i] / denom;
                dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom;
            }

            // Обратный ход
            vmath::Vector<T> x(n);
            x[n - 1] = dp[n - 1];
            for (std::size_t i = n - 2; i < n; --i) {
                x[i] = dp[i] - cp[i] * x[i + 1];
                if (i == 0) break;
            }

            return x;
        }


        //решение систем
        static vmath::Vector<T> solve(const Matrix &A, const vmath::Vector<T> &b,
                                      long double tol = std::numeric_limits<long double>::epsilon()) {
            std::size_t n = A.rows();

            if (A.cols() != n) throw std::invalid_argument("LU must be square");
            if (b.size() != n) throw std::invalid_argument("Incompatible sizes for solve");

            auto [L,U,P] = A.lup_decompose(tol);

            return solve_lup(L, U, P, b);
        }

        static vmath::Vector<T> solve_lup(const Matrix &L, const Matrix &U, const Matrix &P,
                                          const vmath::Vector<T> &b) {
            std::size_t n = U.rows();

            if (U.cols() != n) throw std::invalid_argument("LU must be square");
            if (b.size() != n) throw std::invalid_argument("Incompatible sizes for solve");


            auto b_permutated = P * b;

            vmath::Vector<T> xL(n);
            for (int i = 0; i < n; ++i) {
                T x_i = b_permutated[i];
                for (int j = 0; j < i; ++j) { x_i -= L(i, j) * xL[j]; }
                xL[i] = x_i;
            }

            vmath::Vector<T> x(n);
            for (int i = n - 1; i >= 0; --i) {
                T x_i = xL[i];
                for (int j = n - 1; j > i; --j) { x_i -= U(i, j) * x[j]; }
                x[i] = x_i / U(i, i);
            }

            return x;
        }

        static vmath::Vector<T> solve_Gauss(const Matrix &M,
                                          const vmath::Vector<T> &b) {
            return M.inverse() * b;
        }


        // определитель
        T determinant_via_lup(const long double tol = std::numeric_limits<long double>::epsilon()) const {
            auto [L,U,P] = lup_decompose(tol);
            long double prod = 1.0;
            std::size_t n = rows_;
            for (std::size_t i = 0; i < n; ++i) prod *= (long double) U(i, i);
            return (T) prod;
        }

        //обращение матрицы
        Matrix inverse(long double tol = std::numeric_limits<long double>::epsilon()) const {
            if (rows_ != cols_) throw std::invalid_argument("Inverse requires square matrix");
            std::size_t n = rows_;
            Matrix U = Matrix::concatenate_right(*this, Matrix::identity(n));

            for (std::size_t i = 0; i < n; i++) {
                T pivotValue = T(0);
                std::size_t pivot = i;
                for (std::size_t row = i; row < n; row++) {
                    if (std::abs(U(row, i)) > pivotValue) {
                        pivotValue = std::abs(U(row, i));
                        pivot = row;
                    }
                }

                if (pivotValue <= (T) tol) {
                    throw std::runtime_error("Matrix is singular to working precision");
                }

                if (pivot != i) {
                    U.swap_rows(pivot, i);
                }

                for (std::size_t j = i + 1; j < n; j++) {
                    T mult = U(j, i) / U(i, i);
                    for (std::size_t k = i; k < 2 * n; k++) {
                        U(j, k) -= mult * U(i, k);
                    }
                }
            }

            for (int i = n - 1; i >= 0; --i) {
                T delim = U(i, i);
                for (int k = i; k < 2 * n; ++k) {
                    U(i, k) /= delim;
                }
                for (int j = i - 1; j >= 0; --j) {
                    T mult = U(j, i);
                    for (int k = i; k < 2 * n; ++k) {
                        U(j, k) -= U(i, k) * mult;
                    }
                }
            }

            return U.cut(0, n, n, 2 * n);
        }

        // Печать
        friend std::ostream &operator<<(std::ostream &os, const Matrix &m) {
            for (std::size_t i = 0; i < m.rows_; ++i) {
                os << "[";
                for (std::size_t j = 0; j < m.cols_; ++j) {
                    os << m.at_unchecked(i, j);
                    if (j + 1 != m.cols_) os << ", ";
                }
                os << "]\n";
            }
            return os;
        }

    private:
        std::size_t rows_ = 0, cols_ = 0;
        std::vector<T> data_;
    };
}


#endif //MATRIXMATH_H
