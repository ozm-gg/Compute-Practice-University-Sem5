//
// Created by dmits on 08.09.2025.
//

#ifndef VECTORMATH_H
#define VECTORMATH_H



#include <vector>
#include <initializer_list>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace vmath {
    template <typename T>
    class Vector {
        static_assert(std::is_arithmetic<T>::value, "Vector requires arithmetic type");

    public:
        // Конструкторы
        Vector() = default;
        explicit Vector(std::size_t n) : data_(n) {}
        Vector(std::size_t n, const T &value) : data_(n, value) {}
        Vector(std::initializer_list<T> il) : data_(il) {}

        template <class It>
        Vector(It first, It last) : data_(first, last) {}

        // Размер и доступ
        std::size_t size() const noexcept { return data_.size(); }
        bool empty() const noexcept { return data_.empty(); }

        T &operator[](std::size_t i) noexcept { return data_[i]; }
        const T &operator[](std::size_t i) const noexcept { return data_[i]; }

        T &at(std::size_t i) { return data_.at(i); }
        const T &at(std::size_t i) const { return data_.at(i); }

        T *data() noexcept { return data_.data(); }
        const T *data() const noexcept { return data_.data(); }

        // Итераторы
        auto begin() noexcept { return data_.begin(); }
        auto end() noexcept { return data_.end(); }
        auto begin() const noexcept { return data_.begin(); }
        auto end() const noexcept { return data_.end(); }

        // Преобразования
        std::vector<T> to_std_vector() const { return data_; }
        static Vector from_std_vector(const std::vector<T> &v) { return Vector(v.begin(), v.end()); }

        // Арифметические операторы
        Vector &operator+=(const Vector &other) {
            check_same_size(other);
            for (std::size_t i = 0; i < size(); ++i) data_[i] += other.data_[i];
            return *this;
        }

        Vector &operator-=(const Vector &other) {
            check_same_size(other);
            for (std::size_t i = 0; i < size(); ++i) data_[i] -= other.data_[i];
            return *this;
        }

        Vector &operator*=(const T &scalar) noexcept {
            for (auto &x : data_) x *= scalar;
            return *this;
        }

        Vector &operator/=(const T &scalar) {
            if (scalar == T(0)) throw std::domain_error("Division by zero");
            for (auto &x : data_) x /= scalar;
            return *this;
        }

        friend Vector operator+(Vector a, const Vector &b) { a += b; return a; }
        friend Vector operator-(Vector a, const Vector &b) { a -= b; return a; }
        friend Vector operator-(Vector a) { for (auto &x : a.data_) x = -x; return a; }

        friend Vector operator*(Vector a, const T &s) { a *= s; return a; }
        friend Vector operator*(const T &s, Vector a) { a *= s; return a; }
        friend Vector operator/(Vector a, const T &s) { a /= s; return a; }

        bool operator==(const Vector &other) const noexcept { return data_ == other.data_; }
        bool operator!=(const Vector &other) const noexcept { return !(*this == other); }

        // Скалярное произведение
        T dot(const Vector &other) const {
            check_same_size(other);
            return std::inner_product(data_.begin(), data_.end(), other.data_.begin(), T(0));
        }

        // Квадрат нормы и норма
        T norm_squared() const noexcept {
            return std::inner_product(data_.begin(), data_.end(), data_.begin(), T(0));
        }

        long double norm() const noexcept {
            return std::sqrt((long double)norm_squared());
        }

        // Возвращает нормализованный вектор (копия). Бросает при нулевой норме.
        Vector normalized() const {
            long double n = norm();
            if (n == 0) throw std::domain_error("Cannot normalize zero vector");
            Vector res = *this;
            for (auto &x : res.data_) x = static_cast<T>(x / n);
            return res;
        }

        // Проекция этого вектора на other
        Vector projection_onto(const Vector &other) const {
            check_same_size(other);
            T denom = other.norm_squared();
            if (denom == T(0)) throw std::domain_error("Projection onto zero vector");
            T scalar = dot(other) / denom;
            return other * scalar;
        }

        // Векторное произведение (только для 3D)
        Vector cross(const Vector &other) const {
            if (size() != 3 || other.size() != 3) throw std::domain_error("Cross product is defined for 3D vectors only");
            return Vector{
                data_[1] * other.data_[2] - data_[2] * other.data_[1],
                data_[2] * other.data_[0] - data_[0] * other.data_[2],
                data_[0] * other.data_[1] - data_[1] * other.data_[0]
            };
        }

        // Эвклидово расстояние между векторами
        long double distance_to(const Vector &other) const {
            check_same_size(other);
            long double sum = 0;
            for (std::size_t i = 0; i < size(); ++i) {
                long double d = static_cast<long double>(data_[i] - other.data_[i]);
                sum += d * d;
            }
            return std::sqrt(sum);
        }

        // Прочие полезные операции
        void fill(const T &value) { std::fill(data_.begin(), data_.end(), value); }
        void push_back(const T &value) { data_.push_back(value); }
        void resize(std::size_t n) { data_.resize(n); }

        // Печать
        friend std::ostream &operator<<(std::ostream &os, const Vector &v) {
            os << "[";
            for (std::size_t i = 0; i < v.size(); ++i) {
                os << v.data_[i];
                if (i + 1 != v.size()) os << ", ";
            }
            os << "]";
            return os;
        }

    private:
        std::vector<T> data_;

        void check_same_size(const Vector &other) const {
            if (size() != other.size()) throw std::length_error("Vector sizes do not match");
        }
    };
}



#endif //VECTORMATH_H
