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
#include <concepts>
#include <iostream>
#include <vector>
namespace matrix
{
template<typename T>
concept TableConcept = requires(T t) {
                           t[0];
                           static_cast<size_t>(t.size());
                       };
// Test case for TableConcept
static_assert(TableConcept<std::vector<int>>);

template<typename T>
concept DoubleTableConcept = requires(T t) {
                                 t[0][0];
                                 static_cast<size_t>(t.size());
                                 static_cast<size_t>(t[0].size());
                             };
// Test case for TableConcept
static_assert(DoubleTableConcept<std::vector<std::vector<int>>>);

template<typename T>
concept NumberConcept = requires(T t) {
                            t += (int)0; // Basic operators
                            t -= (int)0;
                            t *= (int)1;
                            t /= (int)1;
                            t = (int)0; // Assignment with an int
                        };
// Test case for NumberConcept
static_assert(NumberConcept<int>);
static_assert(NumberConcept<float>);
}