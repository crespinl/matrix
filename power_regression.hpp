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

// y = a * x ^ b

template<NumberConcept T>
class PowerRegression : public Regression<T>
{
public:
    PowerRegression(std::vector<Coordinate<T>> const& data)
        : Regression<T>(data)
        , m_a(0)
        , m_b(0)
    { }

    void calculate_model() override
    {
        T n = this->m_data.size();
        T b_numerator = n * this->sum_with_operation([](Coordinate<T> const& c) { return std::log(c.x()) * std::log(c.y()); });
        b_numerator -= (this->sum_with_operation([](Coordinate<T> const& c) { return std::log(c.x()); }) * this->sum_with_operation([](Coordinate<T> const& c) { return std::log(c.y()); }));
        T b_denominator = n * this->sum_with_operation([](Coordinate<T> const& c) { return std::pow(std::log(c.x()), 2); });
        b_denominator -= std::pow(this->sum_with_operation([](Coordinate<T> const& c) { return std::log(c.x()); }), 2);
        m_b = b_numerator / b_denominator;
        m_a = this->sum_with_operation([](Coordinate<T> const& c) { return std::log(c.y()); }) / n;
        m_a -= (m_b / n) * this->sum_with_operation([](Coordinate<T> const& c) { return std::log(c.x()); });
        m_a = std::exp(m_a);
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
        std::cout << m_a << std::endl;
        std::cout << m_b << std::endl;
    }

    T predict(T const& v) const override
    {
        return m_a * std::pow(m_b, v);
    }

    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
        PowerRegression<double> pr { { { 2, 16 }, { 1, 2 }, { 3, 54 }, { 4, 128 } } };
        pr.calculate_model();
        assert_true(likely_equals<double>(pr.a(), 2.0000000000000009) && likely_equals<double>(pr.b(), 2.9999999999999996), "PowerRegression doesn't work for a trivial test");
    }

protected:
    T m_a;
    T m_b;
};