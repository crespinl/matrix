/*
Matrix
Copyright (C) 2026  Louis Crespin

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
#include "lm_regression.hpp"

namespace matrix
{
template<NumberConcept T>
class ExponentialSaturationRegression : public LMRegression<T>
{
    // y = A * (1 - exp(-x / B))
public:
    ExponentialSaturationRegression(std::vector<Coordinate<T>> const& data)
        : LMRegression<T>(data)
        , m_a(0)
        , m_b(0)
    { }

    virtual T predict(T const& v) const override
    {
        return m_a * (1 - std::exp(-v / m_b));
    }

    T a() const
    {
        return m_a;
    }

    T b() const
    {
        return m_b;
    }

    void guess_parameters(std::optional<T> a, std::optional<T> b)
    {
        m_approx_a = a;
        m_approx_b = b;
    }

    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE

        // Synthetic curve near y = 10 * (1 - exp(-x/2))
        ExponentialSaturationRegression<long double> r {
            std::vector<Coordinate<long double>>{
                { 0.0L, 0.0L },
                { 1.0L, 3.9347L },
                { 2.0L, 6.3212L },
                { 3.0L, 7.7687L },
                { 4.0L, 8.6466L },
                { 5.0L, 9.1792L }
            }
        };
        r.calculate_model();

        assert_true(std::abs(r.a() - 10.0L) < 0.05L, "ExponentialSaturation regression is broken (A)");
        assert_true(std::abs(r.b() - 2.0L) < 0.05L, "ExponentialSaturation regression is broken (B)");
        assert_true(r.stats().r2 > 0.999L, "ExponentialSaturation regression is not precise enough");
    }

private:
    virtual void apply(Matrix<T> const& p) override
    {
        m_a = p(0, 0);
        m_b = p(0, 1);
    }

    virtual Matrix<T> initials_parameters() const override
    {
        std::optional<T> a = m_approx_a;
        std::optional<T> b = m_approx_b;

        // Safe defaults:
        // A ~= max observed y (or 1 if unavailable)
        // B ~= 1 (user can override through guess_parameters)
        if (!a.has_value())
        {
            if (!this->m_data.empty())
            {
                T max_y = this->m_data[0].y();
                for (auto const& c : this->m_data)
                {
                    if (c.y() > max_y)
                    {
                        max_y = c.y();
                    }
                }
                a = (max_y == static_cast<T>(0)) ? static_cast<T>(1) : max_y;
            }
            else
            {
                a = static_cast<T>(1);
            }
        }

        return { std::vector<std::vector<T>>{
            { a.value_or(static_cast<T>(1)) },
            { b.value_or(static_cast<T>(1)) }
        } };
    }

    virtual Matrix<T> compute_jacobian_matrix(Matrix<T> const& p) const override
    {
        // f(x;A,B) = A * (1 - exp(-x/B))
        // df/dA = 1 - exp(-x/B)
        // df/dB = -A * exp(-x/B) * (x / B^2)

        size_t n = this->m_data.size();
        Matrix<T> r { n, 2 };

        T A = p(0, 0);
        T B = p(0, 1);

        for (size_t i = 0; i < n; i++)
        {
            T x = this->m_data[i].x();
            T e = std::exp(-x / B);

            r(i, 0) = 1 - e;
            r(i, 1) = -A * e * (x / (B * B));
        }

        return r;
    }

    virtual T predict_generic(Matrix<T> const& m, T const& v) const override
    {
        if (m.height() < 2)
        {
            throw Error(Error::Type::wrong_number_of_arguments_in_predict);
        }
        return m(0, 0) * (1 - std::exp(-v / m(0, 1)));
    }

    virtual Matrix<T> diag(Matrix<T> const& m) const override
    {
        return Matrix<T>::get_identity(m.height());
    }

    T m_a;
    T m_b;

    // Useful if the user has any idea of the values
    std::optional<T> m_approx_a;
    std::optional<T> m_approx_b;
};
}
