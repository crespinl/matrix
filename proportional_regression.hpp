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

#include "polynomial_regression.hpp"
#include "tests.hpp"
#include <cmath>
#include <iomanip>
namespace matrix
{
template<NumberConcept T>
class ProportionalRegression : public PolynomialRegression<T>
{
public:
    ProportionalRegression(std::vector<Coordinate<T>> const& data)
        : PolynomialRegression<T>(data)
    { }

    void calculate_model() override
    {
        PolynomialRegression<T>::calculate_coef(1, true);
    }
    T a() const
    {
        return this->m_coef[0];
    }

    void display() const
    {
        std::cout << a() << std::endl;
    }
    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
        ProportionalRegression<double> pr { { { 1, 1 }, { 3, 4 }, { 3, 5 } } };
        pr.calculate_model();
        assert_true(likely_equals(pr.a(), 1.4736842105263), "ProportionalRegression doesn't work for a trivial test");
        assert_true(likely_equals(pr.predict(1), 1.473684210526315708), "ProportionalRegression::predict is broken");
        // A test for Regression that is easier to do here
        ProportionalRegression<long double> pr2 {{ { 1, 2 }, { 2, 4 }, { 4, 9 }, { 5, 15 } }};
        pr2.calculate_model();
        auto stats = pr2.stats();
        assert_true(likely_equals(stats.r2, 0.92359018510546709), "R2 calculation in Regression::stats is broken");
    }

private:
};
}