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

#include "polynomial_regression.hpp"
#include "tests.hpp"
#include <cmath>
#include <iomanip>
namespace Matrix
{
template<NumberConcept T>
class LinearRegression : public PolynomialRegression<T>
{
public:
    LinearRegression(std::vector<Coordinate<T>> const& data)
        : PolynomialRegression<T>(data)
    { }

    void calculate_model() override
    {
        PolynomialRegression<T>::calculate_coef(1);
    }
    T a() const
    {
        return this->m_coef[1];
    }
    T b() const
    {
        return this->m_coef[0];
    }

    void display() const
    {
        std::cout << a() << " " << b() << std::endl;
    }
    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
        LinearRegression<double> lr { { { 1, 2 }, { 2, 4 }, { 4, 9 }, { 5, 10 } } };
        lr.calculate_model();
        assert_true(likely_equals(lr.a(), 2.1) && likely_equals(lr.b(), -0.05), "LinearRegression doesn't work for a trivial test");
    }
};
}