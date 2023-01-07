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
#include "linear_regression.hpp"
#include "matrix.hpp"
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

vector<Matrix<double>> read_input_file()
{
    ifstream file { "./test_input.txt" };
    vector<Matrix<double>> r;
    while (file.peek() != EOF)
    {
        size_t n;
        file >> n;
        Matrix<double> m { n };
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double actual;
                file >> actual;
                m(j, i) = actual;
            }
        }
        r.push_back(m);
    }
    return r;
}

int main()
{
    int nb_success = 0;
    int nb_test = 0;
    Matrix<int>::Assert(nb_success, nb_test);
    cout << "Result : " << nb_success << " tests succeded on " << nb_test << " tests" << endl;

    LinearRegression<double> lr{{{1,2},{2,4}, {4,9} ,{5,10}}};
    lr.calculate_model();
    lr.display();
    return 0;
}