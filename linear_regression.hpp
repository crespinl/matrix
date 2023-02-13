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
        //Tests for Regression thar are easier to do here
        auto stats = lr.stats();
        assert_true(likely_equals(stats.x_mean, 3.) && likely_equals(stats.y_mean, 6.25), "The mean is broken in Regression::stats");
        assert_true(likely_equals(stats.x_variance, 2.5) && likely_equals(stats.y_variance, 11.1875), "The variance is broken in Regression::stats");
        assert_true(likely_equals(stats.x_standart_deviation, 1.5811388300841898) && likely_equals(stats.y_standart_deviation, 3.344772040064913), "The mean is broken in Regression::stats");
        assert_true(likely_equals(stats.covariance, 5.25), "The covariance is broken in Regression::stats");
    }
};
}