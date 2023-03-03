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
#include "matrix.hpp"
#include "regression.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
namespace matrix
{
template<NumberConcept T>
class LMRegression : public Regression<T>
{
public:
    LMRegression(std::vector<Coordinate<T>> const& data)
        : Regression<T>(data)
        , m_max_iter(300)
    { }

    void set_max_iter(size_t max_iter = 300)
    {
        m_max_iter = max_iter;
    }

    void calculate_model() override
    {
        // This is the Levenberg-Marquardt algorithm
        T const initial_lambda = 0.01;
        T const lambda_step = 10;
        T const stop_condition = std::numeric_limits<T>::epsilon(); // If the khi2 of the model is smaller than stop_condition, we stop iterating
        int const max_small_khi2_consecutive_changes = 3;
        auto p = initials_parameters();
        try
        {
            T current_khi2 = khi2(p);
            T lambda = initial_lambda;
            int actual_small_khi2_consecutive_changes = 0;
            for (size_t i = 0; i < m_max_iter; i++)
            {
                auto j = compute_jacobian_matrix(p);
                auto j_t = j;
                j_t.transpose();
                auto r = compute_residual_matrix(p);
                auto delta = j_t * j;
                auto di = diag(delta);
                di *= lambda;
                delta += di;
                delta.inverse();
                delta *= j_t;
                delta.transpose();
                delta *= r;

                auto new_p = p - delta;
                T new_khi2 = khi2(new_p);
                if (std::abs(new_khi2 - current_khi2) < stop_condition)
                {
                    actual_small_khi2_consecutive_changes++;
                    if (actual_small_khi2_consecutive_changes > max_small_khi2_consecutive_changes)
                    {
                        break;
                    }
                }
                else
                {
                    actual_small_khi2_consecutive_changes = 0;
                }
                if (new_khi2 < current_khi2) // We have made progress
                {
                    p = new_p;
                    lambda /= lambda_step;
                    current_khi2 = new_khi2;
                    if (current_khi2 < stop_condition) // The model is precise enough
                    {
                        break;
                    }
                }
                else // We were to far
                {
                    lambda *= lambda_step;
                    // Skip the end and recompute parameters with the new lambda
                }
            }
        }
        catch (Error const& e)
        {
            // We can get an error due to float precision. We return directly
        }
        apply(p);
    }

    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
    }

private:
    virtual void apply(Matrix<T> const& p) = 0;
    virtual Matrix<T> initials_parameters() const = 0;
    virtual Matrix<T> compute_jacobian_matrix(Matrix<T> const& p) const = 0;
    Matrix<T> compute_residual_matrix(Matrix<T> const& p)
    {
        size_t n = this->m_data.size();
        Matrix<T> r { 1, n };
        for (size_t i = 0; i < n; i++)
        {
            r(0, i) = predict_generic(p, this->m_data[i].x()) - this->m_data[i].y();
        }
        return r;
    }
    virtual Matrix<T> diag(Matrix<T> const& m) const
    {
        if (!m.is_square())
        {
            throw Error(Error::Type::matrix_must_be_square);
        }
        Matrix<T> r = Matrix<T>::get_identity(m.height());
        r.fill(0);
        for (size_t i = 0; i < m.height(); i++)
        {
            r(i, i) = m(i, i);
        }
        return r;
    }
    virtual T predict_generic(Matrix<T> const& m, T const& v) const = 0;
    T khi2(Matrix<T> const& p)
    {
        return this->sum_with_operation([&](Coordinate<T> const& c) { return std::pow(c.y() - predict_generic(p, c.x()), 2); });
    }
    size_t m_max_iter;
};
}
