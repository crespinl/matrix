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
#include <chrono>
#include <random>
namespace matrix
{
template<NumberConcept T>
class DampedTrigonometricRegression : public LMRegression<T>
{
    // y = a * sin(b*x+c) * exp(d*x)
public:
    DampedTrigonometricRegression(std::vector<Coordinate<T>> const& data)
        : LMRegression<T>(data)
        , m_a(0)
        , m_b(0)
        , m_c(0)
        , m_d(0)
    {
        std::sort(this->m_data.begin(), this->m_data.end(), [](Coordinate<T> const& c1, Coordinate<T> const& c2) { return c1.x() < c2.x(); });
        // Points need to be sorted here for the approximation algorithm
    }

    virtual T predict(T const& v) const override
    {
        return m_a * std::sin((m_b * v) + m_c) * std::exp(m_d * v);
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
    T d() const
    {
        return m_d;
    }
    void guess_parameters(std::optional<T> a, std::optional<T> b, std::optional<T> c, std::optional<T> d)
    {
        m_approx_a = a;
        m_approx_b = b;
        m_approx_c = c;
        m_approx_d = d;
    }

    static void Assert(int& nb_success, int& nb_test)
    {
        CREATE_ASSERT_TRUE
        std::vector<Coordinate<long double>> points;
        auto f = [](long double v) { return 4. * std::sin(1.5 * v + 0.8) * std::exp(-0.2 * v); };
        for (size_t i = 0; i < 20; i++)
        {
            points.push_back(Coordinate<long double> { static_cast<long double>(i) / 2., f(static_cast<long double>(i) / 2.) });
        }

        DampedTrigonometricRegression<long double> test { points };
        test.calculate_model();
        assert_true(likely_equals(test.a(), 4.0l) && likely_equals(test.b(), 1.5l) && likely_equals(test.c(), 0.8l) && likely_equals(test.d(), -0.2l), "DampedTrigonometricRegression is broken");
        assert_true(likely_equals(test.stats().r2, 1.), "DampedTrigonometricRegression should be more precise");
    }

private:
    virtual void apply(Matrix<T> const& p) override
    {
        m_a = p(0, 0);
        m_b = p(0, 1);
        m_c = mod(p(0, 2), 2 * std::numbers::pi_v<T>); // C is defined %2*pi
        m_d = p(0, 3);
    }
    virtual Matrix<T> initials_parameters() const override
    {
        std::optional<T> a = m_approx_a;
        std::optional<T> b = m_approx_b;
        std::optional<T> c = m_approx_c;
        std::optional<T> d = m_approx_d;
        // Normally, we use the integral equation method, wich is verry precise but can fail
        initials_parameters_with_integral_equation(a, b, c, d);
        return { std::vector<std::vector<T>> { { a.value_or(1) }, { b.value_or(1) }, { c.value_or(1) }, { d.value_or(1) } } };
    }
    virtual Matrix<T> compute_jacobian_matrix(Matrix<T> const& p) const override
    {
        size_t n = this->m_data.size();
        Matrix<T> r { n, 4 };
        for (size_t i = 0; i < n; i++)
        {
            T a = p(0, 0);
            T b = p(0, 1);
            T c = p(0, 2);
            T d = p(0, 3);
            T x = this->m_data[i].x();
            r(i, 0) = std::sin(b * x + c) * std::exp(d * x);
            r(i, 1) = a * x * std::cos(b * x + c) * std::exp(d * x);
            r(i, 2) = a * std::cos(b * x + c) * std::exp(d * x);
            r(i, 3) = a * std::sin(b * x + c) * x * std::exp(d * x);
        }
        return r;
    }
    virtual T predict_generic(Matrix<T> const& m, T const& v) const override
    {
        if (m.height() < 4)
        {
            throw Error(Error::Type::wrong_number_of_arguments_in_predict);
        }
        return m(0, 0) * std::sin(m(0, 1) * v + m(0, 2)) * std::exp(m(0, 3) * v);
    }

public:
    void initials_parameters_with_integral_equation(std::optional<T>& a, std::optional<T>& b, std::optional<T>& c, std::optional<T>& d) const
    /*
    This function computes an approximation for the parameters using an algorithm based on integral equations.
    */
    {
        size_t const n = this->m_data.size();
        std::vector<T> S;
        S.reserve(n);
        S.push_back(0);
        for (size_t i = 1; i < n; i++)
        {
            S.push_back(S[i - 1] + 0.5 * (this->m_data[i].y() + this->m_data[i - 1].y()) * (this->m_data[i].x() - this->m_data[i - 1].x()));
        }

        std::vector<T> SS;
        SS.reserve(n);
        SS.push_back(0);
        for (size_t i = 1; i < n; i++)
        {
            SS.push_back(SS[i - 1] + 0.5 * (S[i] + S[i - 1]) * (this->m_data[i].x() - this->m_data[i - 1].x()));
        }

        // then a little bit of sums ...
        T x_sum = this->sum_with_operation([](Coordinate<T> const& c) { return c.x(); });
        T y_sum = this->sum_with_operation([](Coordinate<T> const& c) { return c.y(); });
        T square_x_sum = this->sum_with_operation([](Coordinate<T> const& c) { return std::pow<T>(c.x(), 2); });
        T x_times_y_sum = this->sum_with_operation([](Coordinate<T> const& c) { return c.x() * c.y(); });
        T SS_sum = 0;
        T square_SS_sum = 0;
        for (auto const& e : SS)
        {
            SS_sum += e;
            square_SS_sum += std::pow<T>(e, 2);
        }
        T y_times_SS_sum = 0;
        T SS_times_S_sum = 0;
        T SS_times_x_sum = 0;
        T SS_times_y_sum = 0;
        T square_S_sum = 0;
        T S_times_x_sum = 0;
        T S_sum = 0;
        T S_times_y_sum = 0;
        for (size_t i = 0; i < n; i++)
        {
            y_times_SS_sum += this->m_data[i].y() * SS[i];
            SS_times_S_sum += SS[i] * S[i];
            SS_times_x_sum += SS[i] * this->m_data[i].x();
            SS_times_y_sum += SS[i] * this->m_data[i].y();
            square_S_sum += std::pow<T>(S[i], 2);
            S_times_x_sum += S[i] * this->m_data[i].x();
            S_sum += S[i];
            S_times_y_sum += S[i] * this->m_data[i].y();
        }
        Matrix<T> m1 { { { square_SS_sum, SS_times_S_sum, SS_times_x_sum, SS_sum },
            { SS_times_S_sum, square_S_sum, S_times_x_sum, S_sum },
            { SS_times_x_sum, S_times_x_sum, square_x_sum, x_sum },
            { SS_sum, S_sum, x_sum, static_cast<T>(n) } } };
        Matrix<T> m2 { { { SS_times_y_sum },
            { S_times_y_sum },
            { x_times_y_sum },
            { y_sum } } };
        try
        {
            m1.inverse();
        }
        catch (...)
        {
            return;
        }
        Matrix<T> r = m1 * m2;
        T d1 = r(0, 1) / 2;
        T b1 = std::sqrt(-(r(0, 0) + std::pow<T>(d1, 2)));

        std::vector<T> beta;
        beta.reserve(n);
        std::vector<T> etha;
        etha.reserve(n);
        for (size_t i = 0; i < n; i++)
        {
            beta.push_back(std::sin(b1 * this->m_data[i].x()) * std::exp(d1 * this->m_data[i].x()));
            etha.push_back(std::cos(b1 * this->m_data[i].x()) * std::exp(d1 * this->m_data[i].x()));
        }
        T square_beta_sum = 0;
        T etha_beta_sum = 0;
        T square_etha_sum = 0;
        T beta_times_y_sum = 0;
        T etha_times_y_sum = 0;
        for (size_t i = 0; i < n; i++)
        {
            square_beta_sum += std::pow<T>(beta[i], 2);
            square_etha_sum += std::pow<T>(etha[i], 2);
            etha_beta_sum += beta[i] * etha[i];
            beta_times_y_sum += beta[i] * this->m_data[i].y();
            etha_times_y_sum += etha[i] * this->m_data[i].y();
        }
        Matrix<T> m3 { { { square_beta_sum, etha_beta_sum },
            { etha_beta_sum, square_etha_sum } } };
        Matrix<T> m4 { { { beta_times_y_sum },
            { etha_times_y_sum } } };
        try
        {
            m3.inverse();
        }
        catch (...)
        {
            return;
        }
        auto r2 = m3 * m4;
        T a1 = std::sqrt(std::pow<T>(r2(0, 0), 2) + std::pow<T>(r2(0, 1), 2));
        T c1;
        if (r2(0, 0) > 0)
        {
            c1 = std::atan(r2(0, 1) / r2(0, 0));
        }
        else
        {
            c1 = std::atan(r2(0, 1) / r2(0, 0)) + std::numbers::pi_v<T>;
        }
        if (!a)
            a = a1;
        if (!b)
            b = b1;
        if (!c)
            c = c1;
        if (!d)
            d = d1;
    }

private:
    static T mod(T const& a, T const& b)
    {
        if (a > 0)
        {
            return fmod(a, b);
        }
        else if (a == 0)
        {
            return 0;
        }
        else
        {
            return b - fmod(-a, b);
        }
    }
    T m_a;
    T m_b;
    T m_c;
    T m_d;
    // Usefull if the user has any idea of the values
    std::optional<T> m_approx_a;
    std::optional<T> m_approx_b;
    std::optional<T> m_approx_c;
    std::optional<T> m_approx_d;
};
}
