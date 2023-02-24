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
class LogisticRegression : public LMRegression<T>
{
    // y = c / (1+a*exp(-b*x))
public:
    LogisticRegression(std::vector<Coordinate<T>> const& data)
        : LMRegression<T>(data)
        , m_a(0)
        , m_b(0)
        , m_c(0)
    { }

    virtual T predict(T const& v) const override
    {
        return m_c / (1 + m_a * std::exp(-m_b * v));
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
        LogisticRegression<long double> lr { std::vector<Coordinate<long double>> { { 0, 0 }, { 1, 0.1 }, { 2, 0.1 }, { 3, 5 }, { 4, 9.9 }, { 5, 10 } } };
        lr.calculate_model();
        LogisticRegression<long double> lr2 { std::vector<Coordinate<long double>> { { 0, 0 }, { 1, 2 }, { 2, 4 }, { 3, 4.1 }, { 4, 4.1 }, { -1, -0.1 } } };
        lr2.calculate_model();
        assert_true(std::abs(lr.a() -  870454.20) < 0.01, "Logistic regression is broken (a)");
        assert_true(std::abs(lr.b() -  4.559) < 0.001, "Logistic regression is broken (b)");
        assert_true(std::abs(lr.c() -  10.002) < 0.001, "Logistic regression is broken (c)");
        assert_true(lr.stats().r2 > 0.9999, "Logistic regression is not precise enough");

        assert_true(std::abs(lr2.a() -  102.27) < 0.01, "Logistic regression is broken (2a)");
        assert_true(std::abs(lr2.b() -  4.585) < 0.001, "Logistic regression is broken (2b)");
        assert_true(std::abs(lr2.c() -  4.082) < 0.001, "Logistic regression is broken (2c)");
        assert_true(lr2.stats().r2 > 0.9993, "Logistic regression 2 is not precise enough");
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
        std::optional<T> a = m_approx_a;
        std::optional<T> b = m_approx_b;
        std::optional<T> c = m_approx_c;
        return { std::vector<std::vector<T>> { { a.value_or(1) }, { b.value_or(1) }, { c.value_or(1) } } };
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
            T d = 1. + a * std::exp(-b * x);

            r(i, 0) = (-c * std::exp(-b * x)) / (d * d);
            r(i, 1) = (c * a * x * std::exp(-b * x)) / (d * d);
            r(i, 2) = 1. / d;
        }
        return r;
    }
    virtual T predict_generic(Matrix<T> const& m, T const& v) const override
    {
        if (m.height() < 3)
        {
            throw Error(Error::Type::wrong_number_of_arguments_in_predict);
        }
        return m(0, 2) / (1 + m(0, 0) * std::exp(-m(0, 1) * v));
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
