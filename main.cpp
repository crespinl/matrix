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
#include "coordinate.hpp"
#include "exponential_regression.hpp"
#include "exponential_regression_2.hpp"
#include "fourier.hpp"
#include "linear_regression.hpp"
#include "logarithmic_regression.hpp"
#include "logistic_regression.hpp"
#include "matrix.hpp"
#include "polynomial_regression.hpp"
#include "power_regression.hpp"
#include "proportional_regression.hpp"
#include "trigonometric_regression.hpp"
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;
using namespace matrix;

int main()
{
    int nb_success = 0;
    int nb_test = 0;
    cout << setprecision(17);
    Matrix<int>::Assert(nb_success, nb_test);
    LinearRegression<int>::Assert(nb_success, nb_test);
    ProportionalRegression<int>::Assert(nb_success, nb_test);
    PolynomialRegression<int>::Assert(nb_success, nb_test);
    ExponentialRegression<int>::Assert(nb_success, nb_test);
    ExponentialRegression2<int>::Assert(nb_success, nb_test);
    LogarithmicRegression<int>::Assert(nb_success, nb_test);
    PowerRegression<int>::Assert(nb_success, nb_test);
    matrix::Fourier::Assert(nb_success, nb_test);
    TrigonometricRegression<int>::Assert(nb_success, nb_test);
    LogisticRegression<int>::Assert(nb_success, nb_test);
    cout << "Result : " << nb_success << " tests succeded on " << nb_test << " tests" << endl;

    return 0;
}