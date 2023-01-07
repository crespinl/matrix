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

#include "regression.hpp"
#include "tests.hpp"
#include <cmath>
#include <iomanip>

template<NumberConcept T>
class LinearRegression : public Regression<T>
{
public:
    LinearRegression(std::vector<Coordinate<T>> const& data)
        : Regression<T>(data)
        , m_a(0)
        , m_b(0)
    { }

    void calculate_model() override
    {
        Regression<T>::calculate_averages();
        T numerator_1 = 0;
        for (auto& e : this->m_data)
        {
            numerator_1 += e.x() * e.y();
        }
        T numerator_2 = this->m_data.size() * this->m_avg_x * this->m_avg_y;
        T denumerator = 0;
        for (auto& e : this->m_data)
        {
            denumerator += std::pow((e.x() - this->m_avg_x), 2);
        }
        m_a = (numerator_1 - numerator_2) / denumerator;
        m_b = this->m_avg_y - m_a * this->m_avg_x;
        T x_sum_sqares = 0;
        for (auto& e : this->m_data)
        {
            x_sum_sqares += std::pow(e.x() - this->m_avg_x, 2);
        }
        T y_sum_sqares = 0;
        for (auto& e : this->m_data)
        {
            y_sum_sqares += std::pow(e.y() - this->m_avg_y, 2);
        }
        this->m_r = std::sqrt((std::pow(m_a, 2) * x_sum_sqares) / y_sum_sqares);
    }
    T a() const
    {
        return m_a;
    }
    T b() const
    {
        return m_b;
    }

    void display() const
    {
        std::cout << m_a << " " << m_b << std::endl;
    }
    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
        LinearRegression<double> lr { { { 1, 2 }, { 2, 4 }, { 4, 9 }, { 5, 10 } } };
        lr.calculate_model();
        assert_true(likely_equals(lr.a(), 2.1) && likely_equals(lr.b(), -0.05) && likely_equals(lr.r(), 0.9927108644188), "LinearRegression doesn't work for a trivial test");
    }

private:
    T m_a;
    T m_b;
};