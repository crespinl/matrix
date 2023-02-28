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
#include "lm_regression.hpp"
namespace matrix
{
template<NumberConcept T>
class GaussianRegression : public LMRegression<T>
{
    // y = a * exp(-b * (x + c)^2)
public:
    GaussianRegression(std::vector<Coordinate<T>> const& data)
        : LMRegression<T>(data)
        , m_a(0)
        , m_b(0)
        , m_c(0)
    { }

    virtual T predict(T const& v) const override
    {
        return m_a * std::exp(-m_b * std::pow(v + m_c, 2));
    }
    T a() const
    {
        return m_a;
    }
    T b() const
    {
        return m_b;
    }
    T c() const
    {
        return m_c;
    }
    void guess_parameters(std::optional<T> a, std::optional<T> b, std::optional<T> c)
    {
        m_approx_a = a;
        m_approx_b = b;
        m_approx_c = c;
    }

    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
        GaussianRegression<long double> gr { std::vector<Coordinate<long double>> { { -5, 0 }, { -4, 0.27 }, { -3.5, 1.21 }, { -3, 2 }, { -2.5, 1.21 }, { -2, 0.27 }, { -1.5, 0.02 } } };
        gr.calculate_model();
        assert_true(gr.stats().r2 > 0.99999, "Gaussian regression should be more precise");
        assert_true(std::abs(gr.a() - 2.00) < 0.01, "Gaussian regression is broken");
        GaussianRegression<long double> gr2 { std::vector<Coordinate<long double>> { { -2.5, 0.002 }, { -2, 0.02 }, { -1.5, 0.1 }, { -1, 0.37 }, { -0.5, 0.78 }, { 0.5, 0.78 }, { 1, 0.37 } } };
        gr2.calculate_model();
        assert_true(gr2.stats().r2 > 0.9999, "Gaussian regression should be more precise (2)");
        assert_true(std::abs(gr2.a() - 1.00) < 0.01, "Gaussian regression is broken");
    }

private:
    virtual void apply(Matrix<T> const& p) override
    {
        m_a = p(0, 0);
        m_b = p(0, 1);
        m_c = p(0, 2);
    }
    virtual Matrix<T> initials_parameters() const override
    {
        Matrix<T> m(1, 3);
        m(0, 0) = m_approx_a.value_or(1);
        m(0, 1) = m_approx_b.value_or(1);
        m(0, 2) = m_approx_c.value_or(1);
        return m;
    }
    virtual Matrix<T> compute_jacobian_matrix(Matrix<T> const& p) const override
    {
        size_t n = this->m_data.size();
        Matrix<T> r { n, 3 };
        for (size_t i = 0; i < n; i++)
        {
            T a = p(0, 0);
            T b = p(0, 1);
            T c = p(0, 2);
            T x = this->m_data[i].x();
            T exp = std::exp(-b * std::pow(x + c, 2));

            r(i, 0) = exp;
            r(i, 1) = -(a * std::pow(x + c, 2) * exp);
            r(i, 2) = -(2 * a * b * (x + c) * exp);
        }
        return r;
    }
    virtual T predict_generic(Matrix<T> const& m, T const& v) const override
    {
        if (m.height() < 2)
        {
            throw Error(Error::Type::wrong_number_of_arguments_in_predict);
        }
        return m(0, 0) * std::exp(-m(0, 1) * std::pow(v + m(0, 2), 2));
    }
    virtual Matrix<T> diag(Matrix<T> const& m) const override // Use an identity matrix is better for this particular kind of regression
    {
        return Matrix<T>::get_identity(m.height());
    }
    T m_a;
    T m_b;
    T m_c;
    // Usefull if the user has any idea of the values
    std::optional<T> m_approx_a;
    std::optional<T> m_approx_b;
    std::optional<T> m_approx_c;
};
}
