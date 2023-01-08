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
#include <cmath>
#include <limits>

#define ASSERT_THROWS(expr, text)                                             \
    nb_test++;                                                                \
    try                                                                       \
    {                                                                         \
        expr;                                                                 \
        std::cout << "Test " << nb_test << " failed : " << text << std::endl; \
    }                                                                         \
    catch (...)                                                               \
    {                                                                         \
        nb_success++;                                                         \
    }

#define CREATE_ASSERT_TRUE                                                                   \
    auto assert_true = [&nb_success, &nb_test](bool test, std::string error_message) -> void \
    {                                                                                        \
        nb_test++;                                                                           \
        if (test)                                                                            \
        {                                                                                    \
            nb_success++;                                                                    \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            std::cout << "Test " << nb_test << " failed : " << error_message << std::endl;   \
        }                                                                                    \
    };

template<typename T>
bool likely_equals(T a, T b)
{
    return std::abs(a - b) < 1e-8;
}