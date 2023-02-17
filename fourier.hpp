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
#include "coordinate.hpp"
#include "tests.hpp"
#include <numbers>
#include <vector>

namespace matrix
{
namespace Fourier
{
template<NumberConcept T>
std::vector<Coordinate<T>> naive_dft(std::vector<Coordinate<T>> const& data)
{
    std::vector<std::complex<T>> cdata;
    std::vector<std::complex<T>> cr;
    std::vector<Coordinate<T>> r;
    size_t const N = data.size();
    cdata.reserve(N);
    r.resize(N);
    cr.resize(N);
    for (auto const& e : data)
    {
        cdata.push_back(e.to_complex());
    }
    for (size_t k = 0; k < N; k++)
    {
        cr[k] = 0;
        for (size_t n = 0; n < N; n++)
        {
            cr[k] += cdata[n] * std::exp<T>((T)-2 * std::complex<T> { 0, 1 } * std::numbers::pi_v<T> * (T)k * ((T)n / (T)N));
        }
        r[k] = Coordinate<T>::from_complex(cr[k]);
    }
    return r;
}
void Assert(int& nb_success, int& nb_test)
{
    CREATE_ASSERT_TRUE
    auto computed = Fourier::naive_dft(std::vector<Coordinate<double>> { { 0, 0 }, { 1., 0.8 }, { 3., 0.14 }, { 4., -0.75 } });
    std::vector<Coordinate<double>> expected { { 8, 0.19000000000000006 }, { -1.4500000000000006, 2.8599999999999994 }, { -2, 0.089999999999999081 }, { -4.549999999999998, -3.1400000000000015 } };
    assert_true(computed == expected, "The naive implementation of dft is broken");
}
}
}
