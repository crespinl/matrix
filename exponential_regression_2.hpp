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

#include "matrix.hpp"
#include "regression.hpp"
#include "tests.hpp"
#include <cmath>
#include <iomanip>

// y = e^(a + b * x)
namespace matrix
{
template<NumberConcept T>
class ExponentialRegression2 : public Regression<T>
{
public:
    ExponentialRegression2(std::vector<Coordinate<T>> const& data)
        : Regression<T>(data)
        , m_a(0)
        , m_b(0)
    {
        // Actually, we simply remove all values with a null y to be sure we don't crash
        this->apply_filter([](Coordinate<T> const& c) { return c.y() == 0; });
    }

    void calculate_model() override
    {
        T n = this->m_data.size();
        T b_numerator;
        b_numerator = n * this->sum_with_operation([](Coordinate<T> const& c) { return c.x() * std::log(c.y()); });
        b_numerator -= this->sum_with_operation([](Coordinate<T> const& c) { return c.x(); }) * this->sum_with_operation([](Coordinate<T> const& c) { return std::log(c.y()); });
        T b_denominator;
        b_denominator = n * this->sum_with_operation([](Coordinate<T> const& c) { return std::pow(c.x(), 2); });
        b_denominator -= std::pow(this->sum_with_operation([](Coordinate<T> const& c) { return c.x(); }), 2);
        m_b = b_numerator / b_denominator;
        m_a = this->sum_with_operation([](Coordinate<T> const& c) { return std::log(c.y()); }) / n;
        m_a -= (m_b / n) * this->sum_with_operation([](Coordinate<T> const& c) { return c.x(); });
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
        return std::exp(m_a + m_b * v);
    }

    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
        ExponentialRegression2<double> er { { { 0, 2.72 }, { 1, 4.48 }, { 2, 7.39 } } };
        er.calculate_model();
        assert_true(likely_equals<double>(er.a(), 1.000379626572201364) && likely_equals<double>(er.b(), 0.499747927326102015), "ExponentialRegression2 doesn't work for a trivial test");
        ExponentialRegression2<double> zero_y { { { 0, 0 }, { 1, 2.7 }, { 2, 7.3 } } };
        zero_y.calculate_model();
        assert_true(!std::isnan(zero_y.a()), "Null y in ExponentialRegression2 make it crash");
    }

protected:
    T m_a;
    T m_b;
};
}