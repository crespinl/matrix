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
#include "concepts.hpp"
#include <complex>
#include <iostream>
namespace matrix
{
template<NumberConcept T>
class Coordinate
{
public:
    Coordinate(T x, T y)
        : m_x(x)
        , m_y(y)
    { }

    T const& x() const { return m_x; };
    T& x() { return m_x; };
    T const& y() const { return m_y; };
    T& y() { return m_y; };

    std::complex<T> to_complex() const
    {
        return { m_x, m_y };
    }

    void display() const
    {
        std::cout << "(" << m_x << ", " << m_y << ")" << std::endl;
    }

private:
    T m_x;
    T m_y;
};
}