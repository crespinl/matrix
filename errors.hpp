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

#include <stdexcept>
#include <string>
namespace matrix
{
class Error : public std::runtime_error
{
public:
    enum class Type
    {
        matrix_not_inversible,
        access_out_of_range,
        line_number_out_of_range,
        row_number_out_of_range,
        divide_by_zero,
        multiply_matrix_size_not_compatible,
        add_substract_matrix_size_not_compatible,
        too_small_table_to_fill_the_line,
        polynomial_regression_call_calculate_model,
        wrong_number_of_arguments_in_predict,
        matrix_must_be_square,
        table_size_not_valid_for_matrix_instantiation,
    };
    Error(Type t);
    char const* what() const throw() override;
    Type type() const { return m_type; }

private:
    Type m_type;
};
}