/*
Matrix
Copyright (C) 2022  Louis Crespin

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

#include "matrix.hpp"
#include "regression.hpp"
#include "tests.hpp"
#include <cmath>
#include <iomanip>
namespace matrix
{
template<NumberConcept T>
class PolynomialRegression : public Regression<T>
{
public:
    PolynomialRegression(std::vector<Coordinate<T>> const& data)
        : Regression<T>(data)
        , m_coef()
        , m_null_y_intercept(false)
    { }

    void calculate_coef(int degree, bool null_y_intercept = false)
    {
        m_null_y_intercept = null_y_intercept;
        Matrix<double> X { (size_t)(degree + (int)!null_y_intercept), this->m_data.size() };
        for (size_t j = 0; j < X.height(); j++)
        {
            if (!null_y_intercept)
            {
                X(0, j) = 1;
                for (size_t i = 1; i < X.width(); i++)
                {
                    X(i, j) = std::pow(this->m_data[j].x(), i);
                }
            }
            else
            {
                for (size_t i = 0; i < X.width(); i++)
                {
                    X(i, j) = std::pow(this->m_data[j].x(), i + 1);
                }
            }
        }
        Matrix<double> Y { 1, this->m_data.size() };
        for (size_t j = 0; j < Y.height(); j++)
        {
            Y(0, j) = this->m_data[j].y();
        }
        auto transposed = X;
        transposed.transpose();
        Matrix<double> result = transposed * X;
        result.inverse();
        result *= transposed;
        result *= Y;
        m_coef.reserve(result.height());
        for (int i = 0; i < result.height(); i++)
        {
            m_coef.push_back(result(0, i));
        }
    }

    std::vector<T> get_coef() const { return m_coef; }

    void calculate_model() override { }

    void display() const
    {
        for (auto c : m_coef)
        {
            std::cout << c << std::endl;
        }
    }

    T predict(T const& v) const override
    {
        T prediction = 0;
        if (!m_null_y_intercept)
        {
            for (size_t i = 0; i < m_coef.size(); i++)
            {
                prediction += m_coef[i] * std::pow(v, i);
            }
        }
        else
        {
            for (size_t i = 0; i < m_coef.size(); i++)
            {
                prediction += m_coef[i] * std::pow(v, i + 1);
            }
        }
        return prediction;
    }

    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
        PolynomialRegression<double> pr { { { 0, 5 }, { 1, 15 }, { 2, 57 }, { 3, 179 }, { 4, 453 } } };
        pr.calculate_coef(4);
        auto coef = pr.get_coef();
        assert_true(likely_equals<double>(coef[0], 5) && likely_equals<double>(coef[1], 4) && likely_equals<double>(coef[2], 3) && likely_equals<double>(coef[3], 2) && likely_equals<double>(coef[4], 1), "PolynomialRegression doesn't work for a trivial test");
        assert_true(likely_equals<double>(pr.predict(5), 975), "PolygomialRegression::predict is broken");
    }

protected:
    std::vector<T> m_coef; // The coefficient of the lowest degree is stored in first
    bool m_null_y_intercept;
};
}