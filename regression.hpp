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

#include "coordinate.hpp"
#include <iostream>
#include <string>
#include <vector>

template<NumberConcept T>
class Regression
{
public:
    Regression(std::vector<Coordinate<T>> const& data)
        : m_data(data)
    { }
    void display() const
    {
        std::cout << m_a << " " << m_b << std::endl;
    }
    virtual ~Regression() { }
    virtual void calculate_model() = 0;

protected:
    T m_a;
    T m_b;
    std::vector<Coordinate<T>> m_data;
};
