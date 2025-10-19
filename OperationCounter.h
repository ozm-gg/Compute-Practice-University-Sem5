//
// Created by dmits on 23.09.2025.
//

#ifndef OPERATIONCOUNTER_H
#define OPERATIONCOUNTER_H

namespace vmath {
    inline size_t global_ops = 0;

    template<typename T>
    struct OpCounter {
        static_assert(std::is_arithmetic<T>::value, "OpCounter requires arithmetic type");

        T value;

        OpCounter() : value(0) {
        }

        OpCounter(const T &v) : value(v) {
        }

        operator T() const { return value; }

        OpCounter &operator+=(const OpCounter &other) {
            global_ops++;
            value += other.value;
            return *this;
        }

        OpCounter &operator-=(const OpCounter &other) {
            global_ops++;
            value -= other.value;
            return *this;
        }

        OpCounter &operator*=(const OpCounter &other) {
            global_ops++;
            value *= other.value;
            return *this;
        }

        OpCounter &operator/=(const OpCounter &other) {
            global_ops++;
            value /= other.value;
            return *this;
        }

        OpCounter operator-() const { return OpCounter(-value); }


        friend OpCounter operator+(OpCounter a, const OpCounter &b) { return a += b; }
        friend OpCounter operator-(OpCounter a, const OpCounter &b) { return a -= b; }
        friend OpCounter operator*(OpCounter a, const OpCounter &b) { return a *= b; }
        friend OpCounter operator/(OpCounter a, const OpCounter &b) { return a /= b; }

        friend bool operator==(const OpCounter &a, const OpCounter &b) { return a.value == b.value; }
        friend bool operator!=(const OpCounter &a, const OpCounter &b) { return a.value != b.value; }
        friend bool operator<(const OpCounter &a, const OpCounter &b) { return a.value < b.value; }
        friend bool operator<=(const OpCounter &a, const OpCounter &b) { return a.value <= b.value; }
        friend bool operator>(const OpCounter &a, const OpCounter &b) { return a.value > b.value; }
        friend bool operator>=(const OpCounter &a, const OpCounter &b) { return a.value >= b.value; }

        friend bool operator==(const OpCounter &a, T b) { return a.value == b; }
        friend bool operator==(T a, const OpCounter &b) { return a == b.value; }

        friend bool operator<(const OpCounter &a, T b) { return a.value < b; }
        friend bool operator<(T a, const OpCounter &b) { return a < b.value; }

        friend bool operator>(const OpCounter &a, T b) { return a.value > b; }
        friend bool operator>(T a, const OpCounter &b) { return a > b.value; }

        friend bool operator<=(const OpCounter &a, T b) { return a.value <= b; }
        friend bool operator<=(T a, const OpCounter &b) { return a <= b.value; }

        friend bool operator>=(const OpCounter &a, T b) { return a.value >= b; }
        friend bool operator>=(T a, const OpCounter &b) { return a >= b.value; }
    };

    template<typename T>
    OpCounter<T> abs(const OpCounter<T> &x) {
        return OpCounter<T>(std::abs(x.value));
    }

    template<typename T>
    OpCounter<T> sqrt(const OpCounter<T> &x) {
        global_ops++;
        return OpCounter<T>(std::sqrt(x.value));
    }

    inline void reset_ops() { global_ops = 0; }
    inline size_t get_ops() { return global_ops; }
}

#endif //OPERATIONCOUNTER_H
