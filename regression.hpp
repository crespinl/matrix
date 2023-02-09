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
#include <functional>
#include <iostream>
#include <string>
#include <vector>

template<NumberConcept T>
class Regression
{
public:
    Regression(std::vector<Coordinate<T>> const& data)
        : m_data(data)
        , m_avg_x(0)
        , m_avg_y(0)
    { }
    virtual ~Regression() { }
    virtual void calculate_model() = 0;
    virtual T predict(T const& v) const = 0;

protected:
    void calculate_averages()
    {
        for (auto& e : this->m_data)
        {
            m_avg_x += e.x();
            m_avg_y += e.y();
        }
        m_avg_x /= this->m_data.size();
        m_avg_y /= this->m_data.size();
    }
    T sum_with_operation(std::function<T(Coordinate<T> const&)> f)
    {
        T result = 0;
        for (auto& e : this->m_data)
        {
            result += f(e);
        }
        return result;
    }
    void apply_filter(std::function<bool(Coordinate<T> const&)> del) // Usefull to delete problematic values such as zeros in some cases
    {
        for (size_t i = 0; i < m_data.size(); i++)
        {
            if (del(m_data[i]))
            {
                m_data.erase(m_data.begin() + i);
                i--;
            }
        }
    }
    std::vector<Coordinate<T>> m_data;
    T m_avg_x;
    T m_avg_y;
};
