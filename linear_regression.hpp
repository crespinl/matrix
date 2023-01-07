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
        , m_avg_x(0)
        , m_avg_y(0)
    { }

    void calculate_model() override
    {
        calculate_averages();
        T numerator_1 = 0;
        for (auto& e : this->m_data)
        {
            numerator_1 += e.x() * e.y();
        }
        T numerator_2 = this->m_data.size() * m_avg_x * m_avg_y;
        T denumerator = 0;
        for (auto& e : this->m_data)
        {
            denumerator += std::pow((e.x() - m_avg_x), 2);
        }
        this->m_a = (numerator_1 - numerator_2) / denumerator;
        this->m_b = m_avg_y - this->m_a * m_avg_x;
    }
    T a() const
    {
        return this->m_a;
    }
    T b() const
    {
        return this->m_b;
    }
    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
        LinearRegression<double> lr { { { 1, 2 }, { 2, 4 }, { 4, 9 }, { 5, 10 } } };
        lr.calculate_model();
        assert_true(likely_equals(lr.a(), 2.1) && likely_equals(lr.b(), -0.05), "LinearRegression doesn't work for a trivial test " + std::to_string(lr.b()));
    }

private:
    void calculate_averages()
    {
        for (auto& e : this->m_data)
        {
            m_avg_x += e.x();
            m_avg_y += e.y();
        }
        m_avg_x /= this->m_data.size();
        m_avg_y /= this->m_data.size();
    }
    T m_avg_x;
    T m_avg_y;
};