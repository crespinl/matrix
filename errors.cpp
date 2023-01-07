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

#include "errors.hpp"

Error::Error(Error::Type t)
    : std::runtime_error { "" }
    , m_type { t }
{ }

const char* Error::what() const throw()
{
    switch (m_type)
    {
        case Error::Type::access_out_of_range:
        {
            return "Access out of range";
        }
        case Error::Type::matrix_not_inversible:
        {
            return "The matrix is not inversible !";
        }
        case Error::Type::line_number_out_of_range:
        {
            return "Line number out of range";
        }
        case Error::Type::divide_by_zero:
        {
            return "Division by zero";
        }
        case Error::Type::multiply_matrix_size_not_compatible:
        {
            return "Can't multiply matrix whose dimensions are not compatible";
        }
        case Error::Type::add_matrix_size_not_compatible:
        {
            return "Can't add matrix whose size is not the same";
        }
        case Error::Type::too_small_table_to_fill_the_line:
        {
            return "Table too small to fill the line !";
        }
        default:
        {
            return "Unknown error";
        }
    }
}