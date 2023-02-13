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

// y = a * x ^ b
namespace matrix
{
template<NumberConcept T>
class PowerRegression : public Regression<T>
{
public:
    PowerRegression(std::vector<Coordinate<T>> const& data)
        : Regression<T>(data)
        , m_a(0)
        , m_b(0)
    {
        // Actually, we simply remove all values with a null x or y to be sure we don't crash
        this->apply_filter([](Coordinate<T> const& c) { return c.x() == 0 || c.y() == 0; });
    }

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
        return m_a * std::pow(v, m_b);
    }

    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
        PowerRegression<double> pr { { { 2, 16 }, { 1, 2 }, { 3, 54 }, { 4, 128 } } };
        pr.calculate_model();
        assert_true(likely_equals<double>(pr.a(), 2.0000000000000009) && likely_equals<double>(pr.b(), 2.9999999999999996), "PowerRegression doesn't work for a trivial test");
        PowerRegression<double> zero { { { 0, 0 }, { 0, 2.7 }, { 2, 0 }, { 1, 2 }, { 2, 16 } } };
        zero.calculate_model();
        assert_true(!std::isnan(zero.a()), "Null value in PowerRegression make it crash");
        //Test for Regression that is easier to do here
        PowerRegression<long double> pr2 {{ { 1, 2 }, { 2, 4 }, { 4, 9 }, { 5, 15 } }};
        pr2.calculate_model();
        auto stats = pr2.stats();
        assert_true(likely_equals(stats.r, 0.97691239419760723) && likely_equals(stats.r2, 0.95435782593690122), "R/R2 calculation is broken in Regression::stats");
    }

protected:
    T m_a;
    T m_b;
};
}