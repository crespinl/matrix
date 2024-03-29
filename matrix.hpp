/*
Matrix
Copyright (C) 2022-2023  Louis Crespin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

SPDX itentifier : GPL-3.0-or-later
*/
#pragma once
#include "concepts.hpp"
#include "errors.hpp"
#include "tests.hpp"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

/*
The idea of the Matrix class is to provide a template class for manipulating mathematical matrix.
It must throw at each error, such as access out of range, operation not allowed, ...
Important methods must give in comment their temporal complexity, with n = x_max and m = y_max


A Matrix is represented like that :
0      x
  1 2 3
  4 5 6
  7 8 9
y

*/
namespace matrix
{
template<NumberConcept T>
class Matrix
{
public:
    Matrix(size_t x_max, size_t y_max)
        : m_data(new T[x_max * y_max])
        , m_x_max(x_max)
        , m_y_max(y_max)
    { }
    Matrix(size_t max)
        : m_data(new T[max * max])
        , m_x_max(max)
        , m_y_max(max)
    { }
    template<DoubleTableConcept DT>
    Matrix(DT const& t)
        : m_data(new T[t[0].size() * t.size()])
        , m_x_max(t[0].size())
        , m_y_max(t.size())
    {
        for (size_t i = 0; i < m_x_max; i++)
        {
            for (size_t j = 0; j < m_y_max; j++)
            {
                if (t[j].size() <= i)
                {
                    throw Error(Error::Type::table_size_not_valid_for_matrix_instantiation);
                }
                at_unsafe(i, j) = t[j][i]; // Already checked
            }
        }
    }
    Matrix(std::vector<std::vector<T>> const& t)
        : m_data(new T[t[0].size() * t.size()])
        , m_x_max(t[0].size())
        , m_y_max(t.size())
    {
        for (size_t i = 0; i < m_x_max; i++)
        {
            for (size_t j = 0; j < m_y_max; j++)
            {
                if (t[j].size() <= i)
                {
                    throw Error(Error::Type::table_size_not_valid_for_matrix_instantiation);
                }
                at_unsafe(i, j) = t[j][i]; // Already checked
            }
        }
    }
    Matrix(Matrix<T> const& other)
        : m_data(new T[other.m_data_size()])
        , m_x_max(other.m_x_max)
        , m_y_max(other.m_y_max)
    {
        for (size_t i = 0; i < m_data_size(); i++)
        {
            m_data[i] = other.m_data[i];
        }
    }
    Matrix(Matrix<T>&& other)
        : m_data(other.m_data)
        , m_x_max(std::move(other.m_x_max))
        , m_y_max(std::move(other.m_y_max))
    {
        other.m_data = nullptr;
    }
    void fill(T const& value) // O (n*m)
    {
        for (size_t i = 0; i < m_x_max * m_y_max; i++)
        {
            m_data[i] = value;
        }
    }
    bool operator==(Matrix const& other) const // O (n*m)
    {
        if (m_x_max != other.m_x_max || m_y_max != other.m_y_max)
        {
            return false;
        }
        for (size_t i = 0; i < m_data_size(); i++)
        {
            if (m_data[i] != other.m_data[i])
            {
                return false;
            }
        }
        return true;
    }
    bool operator!=(Matrix const&) const = default; // O (n*m)
    T& operator()(size_t x, size_t y)
    {
        return at(x, y);
    };
    T const& operator()(size_t x, size_t y) const
    {
        return at(x, y);
    }
    Matrix<T>& operator=(Matrix<T> const& other) // O (n*m)
    {
        if (this == &other)
        {
            return *this;
        }
        delete[] (m_data);
        m_data = nullptr;
        m_x_max = other.m_x_max;
        m_y_max = other.m_y_max;
        m_data = new T[m_data_size()];
        for (size_t i = 0; i < m_data_size(); i++)
        {
            m_data[i] = other.m_data[i];
        }
        return *this;
    }
    Matrix<T>& operator=(Matrix<T>&& other)
    {
        delete[] (m_data);
        m_data = nullptr;
        m_x_max = other.m_x_max;
        m_y_max = other.m_y_max;
        m_data = other.m_data;
        other.m_data = nullptr;
        return *this;
    }

    ~Matrix()
    {
        if (m_data != nullptr)
        {
            delete[] (m_data);
            m_data = nullptr;
        }
    }

    friend inline Matrix<T> operator+(Matrix<T> const& m1, Matrix<T> const& m2)
    {
        return Matrix<T>::add(m1, m2);
    }

    friend inline Matrix<T> operator-(Matrix<T> const& m1, Matrix<T> const& m2)
    {
        return Matrix<T>::substract(m1, m2);
    }

    friend inline Matrix<T> operator*(Matrix<T> const& m1, Matrix<T> const& m2)
    {
        return Matrix<T>::multiply_matrix(m1, m2);
    }

    friend inline Matrix<T> operator*(Matrix<T> const& m, T const& value)
    {
        return Matrix<T>::multiply_constant(m, value);
    }

    friend inline Matrix<T> operator*(T const& value, Matrix<T> const& m)
    {
        return Matrix<T>::multiply_constant(m, value);
    }

    Matrix<T>& operator+=(Matrix<T> const& other)
    {
        *this = add(*this, other);
        return *this;
    }

    Matrix<T>& operator-=(Matrix<T> const& other)
    {
        *this = substract(*this, other);
        return *this;
    }

    Matrix<T>& operator*=(Matrix<T> const& other)
    {
        *this = multiply_matrix(*this, other);
        return *this;
    }

    Matrix<T>& operator*=(T const& value)
    {
        *this = multiply_constant(*this, value);
        return *this;
    }

    bool is_square() const // O (1)
    {
        return m_x_max == m_y_max;
    }
    bool is_identity() const // O (n*m)
    {
        T const null_value = (T)0;
        T const one_value = (T)1;
        if (!is_square())
        {
            return false;
        }
        for (size_t j = 0; j < m_y_max; j++)
        {
            for (size_t i = 0; i < m_x_max; i++)
            {
                if (i == j)
                {
                    if (at_unsafe(i, j) != one_value) // Already checked
                    {
                        return false;
                    }
                }
                else
                {
                    if (at_unsafe(i, j) != null_value) // Already checked
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    void transpose() // O (n*m)
    {
        auto transposed = Matrix<T> { m_y_max, m_x_max };
        size_t i, j;
#ifdef USE_OPENMP
#    pragma omp parallel for private(i, j) shared(transposed)
#endif
        for (i = 0; i < m_x_max; i++)
        {
            for (j = 0; j < m_y_max; j++)
            {
                transposed.at_unsafe(j, i) = at_unsafe(i, j); // Already checked
            }
        }
        std::swap(*this, transposed);
    }
    static Matrix<T> get_identity(size_t n)
    {
        Matrix<T> m { n };
        m.fill(0);
        for (size_t i = 0; i < n; i++)
        {
            m.at_unsafe(i, i) = 1; // Already checked
        }
        return m;
    }
    void inverse() // O(n^3)
    {
        *this = calculate_inverse(*this);
    }
    template<TableConcept L>
    void set_line(size_t y, L const& line) // O (n)
    {
        if (line.size() < m_x_max)
        {
            throw Error { Error::Type::too_small_table_to_fill_the_line };
        }
        size_t pos = index_of_unsafe(0, y); // We've already check this access
        for (int i = 0; i < m_x_max; i++)
        {
            m_data[pos + i] = line[i];
        }
    }
    size_t height() const { return m_y_max; }
    size_t width() const { return m_x_max; }
    void display() const
    {
        for (size_t j = 0; j < m_y_max; j++)
        {
            std::cout << '|';
            for (size_t i = 0; i < m_x_max; i++)
            {
                std::cout << m_data[index_of_unsafe(i, j)] << '|'; // Access out of bounds is not possible there
            }
            std::cout << std::endl;
        }
    }
    Matrix<T> minor_matrix_at_index(size_t index) const // Usefull to run testcases
    {
        if (m_x_max != m_y_max)
        {
            throw Error { Error::Type::matrix_must_be_square };
        }
        if (index > m_x_max)
        {
            throw Error { Error::Type::access_out_of_range };
        }
        return get_minor_matrix_at_index(index);
    }
    T determinant() const
    {
        if (m_x_max != m_y_max)
        {
            throw Error { Error::Type::matrix_must_be_square };
        }
        return get_determinant(*this);
    }
    bool is_inversible() const
    {
        return determinant() != 0;
    }
    bool is_symmetrical() const
    {
        if (!is_square())
        {
            throw Error { Error::Type::matrix_must_be_square };
        }
        for (size_t i = 0; i < m_x_max; i++)
        {
            for (size_t j = i; j < m_y_max; j++)
            {
                if (at_unsafe(i, j) != at_unsafe(j, i))
                {
                    return false;
                }
            }
        }
        return true;
    }
    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
        Matrix<int> matrix1(2, 2);
        matrix1.fill(0);
        Matrix<int> matrix2(2, 2);
        matrix2.fill(0);
        Matrix<int> matrix3(2, 2);
        matrix3.fill(0);
        matrix3(0, 0) = 1;
        Matrix<int> matrix4(2, 3);
        matrix4.fill(0);
        auto matrix5 = matrix4;
        Matrix<float> identity_matrix(2, 2);
        identity_matrix.fill(0);
        identity_matrix(0, 0) = 1;
        identity_matrix(1, 1) = 1;

        Matrix<int> matrix_to_transpose { 2, 2 };
        matrix_to_transpose(0, 0) = 1;
        matrix_to_transpose(1, 0) = 2;
        matrix_to_transpose(0, 1) = 3;
        matrix_to_transpose(1, 1) = 4;

        matrix_to_transpose.transpose();

        Matrix<int> expected_transposed { 2, 2 };
        expected_transposed(0, 0) = 1;
        expected_transposed(1, 0) = 3;
        expected_transposed(0, 1) = 2;
        expected_transposed(1, 1) = 4;

        Matrix<int> addition1 { 2, 1 };
        addition1(0, 0) = 1;
        addition1(1, 0) = 2;

        Matrix<int> addition2 { 2, 1 };
        addition2(0, 0) = 4;
        addition2(1, 0) = 3;

        addition1 = addition1 + addition2;

        Matrix<int> addition_result { 2, 1 };
        addition_result.fill(5);

        Matrix<int> scalar_product { 2, 1 };
        scalar_product(0, 0) = 1;
        scalar_product(1, 0) = 2;
        scalar_product = scalar_product * 2;

        Matrix<int> scalar_product_result { 2, 1 };
        scalar_product_result(0, 0) = 2;
        scalar_product_result(1, 0) = 4;

        Matrix<int> product1 { 2, 2 };
        product1(0, 0) = 1;
        product1(1, 0) = 2;
        product1(0, 1) = 3;
        product1(1, 1) = 4;

        Matrix<int> product2 { 2, 2 };
        product2(0, 0) = 4;
        product2(1, 0) = 3;
        product2(0, 1) = 2;
        product2(1, 1) = 1;

        product1 = product1 * product2;

        Matrix<int> product_result { 2, 2 };
        product_result(0, 0) = 8;
        product_result(1, 0) = 5;
        product_result(0, 1) = 20;
        product_result(1, 1) = 13;

        Matrix<float> matrix_to_inverse { 2, 2 };
        matrix_to_inverse(0, 0) = 1;
        matrix_to_inverse(1, 0) = 2;
        matrix_to_inverse(0, 1) = 3;
        matrix_to_inverse(1, 1) = 4;
        matrix_to_inverse.inverse();

        Matrix<float> inversed_matrix { 2, 2 };
        inversed_matrix(0, 0) = -2;
        inversed_matrix(1, 0) = 1;
        inversed_matrix(0, 1) = 1.5;
        inversed_matrix(1, 1) = -0.5;

        Matrix<long double> bigger_matrix_to_inverse { std::vector<std::vector<long double>> {
            { 1, 2, 3, 4 },
            { 5, 2, 7, 8 },
            { 9, 1, 8, 7 },
            { 1, 3, 2, 4 } } };
        bigger_matrix_to_inverse.inverse();

        Matrix<long double> bigger_inversed_matrix { { { -21. / 32., 1. / 32., 1. / 8., 3. / 8. },
            { 33. / 32., -29. / 32., 3. / 8., 1. / 8. },
            { 65. / 32., -29. / 32., 3. / 8., -7 / 8. },
            { -13. / 8., 9. / 8., -1. / 2., 1. / 2. } } };
        Matrix<long double> not_inversible_matrix { std::vector<std::vector<long double>> {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 } } };
        Matrix<long double> minor_matrix_result { { { 4, 6 },
            { 7, 9 } } };
        Matrix<long double> determinant { std::vector<std::vector<long double>> {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 8, 7, 9 } } };
        Matrix<long double> symmetrical { std::vector<std::vector<long double>> {
            { 1, 2, 3 },
            { 2, 5, 7 },
            { 3, 7, 9 } } };

        assert_true(matrix1 == matrix2, "Two equals matrix are not considered as equal");
        assert_true(matrix1 != matrix3, "Two different matrix are considered as equal");
        assert_true(matrix1 != matrix4, "Two matrix with different size are considered as equal");
        assert_true(matrix4 == matrix5, "Copy constructor does not make an identical copy if itself");
        assert_true(identity_matrix.is_identity(), "An identity matrix is not considered as identity");
        assert_true(!matrix1.is_identity(), "An non identity matrix is considered as identity");
        assert_true(matrix_to_transpose == expected_transposed, "Error in transpose()");
        assert_true(addition_result == addition1, "Addition fails");
        assert_true(scalar_product_result == scalar_product, "Scalar product fails");
        assert_true(product_result == product1, "Product fails");
        assert_true(Matrix<int>::get_identity(10).is_identity(), "Identity fails");
        ASSERT_THROWS(matrix1(10, 0), "Access out of range does not throws");
        assert_true(matrix_to_inverse == inversed_matrix, "Matrix inversion fails");
        assert_true(bigger_matrix_to_inverse == bigger_inversed_matrix, "Matrix inversion fails");
        ASSERT_THROWS(not_inversible_matrix.inverse(), "Inversion of a not inversable matrix doesn't throws");
        assert_true(not_inversible_matrix.minor_matrix_at_index(1) == minor_matrix_result, "minor_matrix_at_index is broken");
        assert_true(not_inversible_matrix.determinant() == 0., "Not inversible_matrix does not has a null determinant");
        assert_true(determinant.determinant() == -9., "Determinant is broken");
        assert_true(!not_inversible_matrix.is_inversible(), "Not inversible matrix detected as inversible");
        assert_true(!determinant.is_symmetrical(), "Non symmetrical matrix should not be considered as symmetrical");
        assert_true(symmetrical.is_symmetrical(), "Symetrical matrix not considered as symmetrical");

#undef ASSERT_THROWS
    }

private:
    Matrix(size_t x_max, size_t y_max, std::vector<T> const& data)
        : m_x_max(x_max)
        , m_y_max(y_max)
        , m_data(data)
    { }
    size_t index_of_safe(size_t x, size_t y) const
    {
        auto index = y * m_x_max + x;
        if (x >= m_x_max || y >= m_y_max || index >= m_x_max * m_y_max)
        {
            throw Error { Error::Type::access_out_of_range };
        }
        return index;
    }
    size_t index_of_unsafe(size_t x, size_t y) const
    {
        return y * m_x_max + x;
    }
    T& at(size_t x, size_t y)
    {
        return m_data[index_of_safe(x, y)];
    };
    T const& at(size_t x, size_t y) const
    {
        return m_data[index_of_safe(x, y)];
    }
    T& at_unsafe(size_t x, size_t y)
    {
        return m_data[index_of_unsafe(x, y)];
    };
    T const& at_unsafe(size_t x, size_t y) const
    {
        return m_data[index_of_unsafe(x, y)];
    }
    static Matrix<T> add(Matrix<T> const& m1, Matrix<T> const& m2) // O(n*m)
    {
        if (m1.m_x_max != m2.m_x_max || m1.m_y_max != m2.m_y_max)
        {
            throw Error { Error::Type::add_substract_matrix_size_not_compatible };
        }
        Matrix<T> sum { m1.m_x_max, m1.m_y_max };
#ifdef USE_OPENMP
#    pragma omp parallel for
#endif
        for (size_t i = 0; i < m1.m_data_size(); i++)
        {
            sum.m_data[i] = m1.m_data[i] + m2.m_data[i];
        }
        return sum;
    }
    static Matrix<T> substract(Matrix<T> const& m1, Matrix<T> const& m2) // O(n*m)
    {
        if (m1.m_x_max != m2.m_x_max || m1.m_y_max != m2.m_y_max)
        {
            throw Error { Error::Type::add_substract_matrix_size_not_compatible };
        }
        Matrix<T> sub { m1.m_x_max, m1.m_y_max };
#ifdef USE_OPENMP
#    pragma omp parallel for
#endif
        for (size_t i = 0; i < m1.m_data_size(); i++)
        {
            sub.m_data[i] = m1.m_data[i] - m2.m_data[i];
        }
        return sub;
    }
    static Matrix<T> multiply_constant(Matrix<T> const& m, T const& value) // O(n*m)
    {
        Matrix<T> product { m.m_x_max, m.m_y_max };
#ifdef USE_OPENMP
#    pragma omp parallel for
#endif
        for (size_t i = 0; i < m.m_data_size(); i++)
        {
            product.m_data[i] = m.m_data[i] * value;
        }
        return product;
    }
    static Matrix<T> multiply_matrix(Matrix<T> const& m1, Matrix<T> const& m2) // O(n*m*n)
    {
        // Keep in mind that A * B != B * A for matrix !
        if (m1.m_x_max != m2.m_y_max)
        {
            throw Error { Error::Type::multiply_matrix_size_not_compatible };
        }
        Matrix<T> product { m2.m_x_max, m1.m_y_max };
        product.fill(0);
        size_t i, j, k;
#ifdef USE_OPENMP
#    pragma omp parallel for private(i, j, k) shared(product, m1, m2)
#endif
        for (i = 0; i < m1.m_y_max; i++) // We firstly iterate on the y dimension of the first matrix
        {
            for (k = 0; k < m1.m_x_max; k++) // We secondly iterate on the x dimension of the first matrix (= the common dimension)
            {
                for (j = 0; j < m2.m_x_max; j++) // Then we iterate on the x dimension on the second matrix
                {
                    product.at_unsafe(j, i) += m1.at_unsafe(k, i) * m2.at_unsafe(j, k); // Already checked
                }
            }
        }
        return product;
    }
    void exchange_row(size_t r1, size_t r2) // O(n)
    {
        if (r1 >= m_x_max || r2 >= m_x_max)
        {
            throw Error { Error::Type::row_number_out_of_range };
        }
        if (r1 == r2)
        {
            return;
        }
        for (size_t j = 0; j < m_y_max; j++)
        {
            T temp;
            temp = at_unsafe(r1, j); // Already checked
            at_unsafe(r1, j) = at_unsafe(r2, j);
            at_unsafe(r2, j) = temp;
        }
    }
    void exchange_line(size_t l1, size_t l2) // O(n)
    {
        if (l1 >= m_y_max || l2 >= m_y_max)
        {
            throw Error { Error::Type::line_number_out_of_range };
        }
        if (l1 == l2)
        {
            return;
        }
        for (size_t i = 0; i < m_x_max; i++)
        {
            T temp;
            temp = at_unsafe(i, l1); // Already checked
            at_unsafe(i, l1) = at_unsafe(i, l2);
            at_unsafe(i, l2) = temp;
        }
    }
    void substract_line(size_t l1, size_t l2, T const& coef = 1)
    {
        if (l1 >= m_y_max || l2 >= m_y_max)
        {
            throw Error { Error::Type::line_number_out_of_range };
        }
        size_t const begin_1 = index_of_unsafe(0, l1);
        size_t const begin_2 = index_of_unsafe(0, l2);
        for (size_t i = 0; i < m_x_max; ++i)
        {
            m_data[begin_1 + i] -= m_data[begin_2 + i] * coef;
        }
    }
    void divide_line(size_t l1, T const& value)
    {
        if (l1 >= m_y_max)
        {
            throw Error { Error::Type::line_number_out_of_range };
        }
        if (value == 0)
        {
            throw Error { Error::Type::divide_by_zero };
        }
        else if (value == 1)
        {
            return;
        }
#ifdef USE_OPENMP
#    pragma omp parallel for
#endif
        for (size_t i = 0; i < m_x_max; i++)
        {
            at_unsafe(i, l1) /= value;
        }
    }
    static Matrix<T> calculate_inverse(Matrix<T> m) // O(n^3)
    {
        Matrix<T> s = get_identity(m.m_x_max);
        // The idea is to apply always the same changes to the identity matrix to get from it the inverse at the end
        size_t i, j, k;
        for (j = 0; j < m.m_y_max; j++) // We transform the matrix into a triangular matrix
        {
            if (j != m.m_y_max - 1)
            {
                for (i = j; i < m.m_y_max; i++)
                {
                    if (i == j + 1)
                    {
                        continue;
                    }
                    if (m.at_unsafe(j, i) != (T)0 && (m.at_unsafe(j, j + 1) / m.at_unsafe(j, i) * m.at_unsafe(j + 1, i) != m.at_unsafe(j + 1, j + 1))) // if we wont make the future pivot be null
                    {                                                                                                                                  // Already checked access
                        m.exchange_line(i, j);
                        s.exchange_line(i, j);
                        break;
                    }
                }
            }
            T const pivot = m.at_unsafe(j, j); // Already checked
            if (pivot != (T)0)
            {
                if (pivot != (T)1)
                {
                    m.divide_line(j, pivot);
                    s.divide_line(j, pivot);
                }
#ifdef USE_OPENMP
#    pragma omp parallel for
#endif
                for (k = j + 1; k < m.m_y_max; k++)
                {
                    T coef = m.at_unsafe(j, k) / m.at_unsafe(j, j); // Already checked
                    m.substract_line(k, j, coef);
                    s.substract_line(k, j, coef);
                }
            }
            else
            {
                throw Error { Error::Type::matrix_not_inversible };
            }
        }
#ifdef USE_OPENMP
#    pragma omp parallel for
#endif
        for (j = 0; j < m.m_y_max; j++) // Then we make the matrix have only 1 in his diagonal
        {
            m.divide_line(j, m.at_unsafe(j, j)); // Already checked
            s.divide_line(j, m.at_unsafe(j, j));
        }
        for (j = m.m_y_max - 2; j != (size_t)-1; j--) // Then we make the matrix be the identity
        {
            for (k = j; k < m.m_x_max - 1; k++)
            {
                T coef = m.at_unsafe(k + 1, j); // Already checked
                // This is not usefull to modify m here since it will not anymore be read
                s.substract_line(j, k + 1, coef);
            }
        }
        return s;
    }
    // This is private so it can be unsafe. It makes the assumption that the matrix is square, that i is not out of range and that the size is miminum 2*2
    Matrix<T> get_minor_matrix_at_index(size_t index) const
    {
        Matrix<T> minor { m_x_max - 1 };
        size_t i;
        for (i = 0; i < index; i++)
        {
            for (size_t j = 1; j < m_y_max; j++)
            {
                minor.at_unsafe(i, j - 1) = at_unsafe(i, j);
            }
        }
        i++;
        for (; i < m_x_max; i++)
        {
            for (size_t j = 1; j < m_y_max; j++)
            {
                minor.at_unsafe(i - 1, j - 1) = at_unsafe(i, j);
            }
        }
        return minor;
    }
    static bool lup_decompose(Matrix<T>& m, std::vector<size_t>& P)
    {
        // Adapted from Wikipedia
        // Computes an in-place LUP decomposition, where P is stored as list of column indexes of 1 in P
        // The last element of P stores the number of row exchanges that have been made
        size_t N = m.height();
        for (size_t i = 0; i < N + 1; i++)
        {
            P[i] = i;
        }

        for (size_t i = 0; i < N; i++)
        {
            T max_a = 0.;
            size_t i_max = i;

            for (size_t k = i; k < N; k++)
            {
                size_t abs_a = std::fabs(m.at_unsafe(k, i));
                if (abs_a > max_a)
                {
                    max_a = abs_a;
                    i_max = k;
                }
            }

            if (max_a < std::numeric_limits<T>::epsilon())
            {
                return false;
            }

            if (i_max != i)
            {
                std::swap(P[i], P[i_max]);
                m.exchange_row(i, i_max);
                P[N]++;
            }
            for (size_t j = i + 1; j < N; j++)
            {
                m.at_unsafe(j, i) /= m.at_unsafe(i, i);

                for (size_t k = i + 1; k < N; k++)
                {
                    m.at_unsafe(j, k) -= m.at_unsafe(j, i) * m.at_unsafe(i, k);
                }
            }
        }

        return true;
    }
    static T lup_determinant(Matrix<T> const& m, std::vector<size_t> P)
    {
        size_t N = m.height();
        T r = m.at_unsafe(0, 0);

        for (size_t i = 1; i < N; i++)
        {
            r *= m.at_unsafe(i, i);
        }
        return (P[N] - N) % 2 == 0 ? r : -r;
    }
    static T get_determinant(Matrix<T> m) // O(n^3)
    {
        std::vector<size_t> P;
        P.resize(m.height() + 1);
        lup_decompose(m, P);
        return lup_determinant(m, P);
    }
    size_t inline m_data_size() const
    {
        return m_x_max * m_y_max;
    }
    T* m_data;
    size_t m_x_max;
    size_t m_y_max;
};
}