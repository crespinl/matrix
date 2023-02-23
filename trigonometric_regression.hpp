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
class TrigonometricRegression : public LMRegression<T>
{
    // y = a * sin(b*x+c)+d
public:
    TrigonometricRegression(std::vector<Coordinate<T>> const& data)
        : LMRegression<T>(data)
        , m_a(0)
        , m_b(0)
        , m_c(0)
        , m_d(0)
    { }

    virtual T predict(T const& v) const override
    {
        return m_a * std::sin((m_b * v) + m_c) + m_d;
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
    // Since a trigonometric model heavily depends on the initial guess for all parameters, the user could give a guess if he has one
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
        TrigonometricRegression<long double> tr { std::vector<Coordinate<long double>> { { 0, 2 }, { 1., 1.88 }, { 2, 1.54 }, { 3, 1.07 }, { 5, 0.20 }, { 6, 0.01 }, { 7, 0.06 }, { 4, 0.58 }, { 8, 0.35 }, { 9, 0.79 }, { 10, 1.28 } } };
        tr.calculate_model();
        TrigonometricRegression<long double> tr2 { std::vector<Coordinate<long double>> { { 0, 0 }, { 1., 0.8 }, { 3., 0.14 }, { 4., -0.75 } } };
        tr2.calculate_model();
        TrigonometricRegression<long double> tr3 { std::vector<Coordinate<long double>> { { 0, 2.9 }, { 6, 2.29 }, { 12, 0.01 }, { 18, -2.28 }, { 24, -2.9 }, { 36, 1.19 }, { 48, 2.41 }, { 60, -2.15 }, { 64, -2.91 } } };
        tr3.guess_parameters({}, {}, {}, {});
        tr3.calculate_model();
        // Since the values in itselves are not very precise, the tests are quite inprecise
        assert_true(std::abs(tr.a() - 1.00134) < 0.001, "TrigonometricRegression is broken (test 1)"); // All parameters depends on each other, so testing one should be enough
        assert_true(tr.stats().r2 >= 0.9999, "The R2 value should be bigger in TrigonometricRegression (test 1)");
        assert_true(tr2.stats().r2 == 1, "The R2 value should be bigger in TrigonometricRegression (test 2)");
        assert_true(std::abs(tr3.b() - 0.148) < 0.02, "Values not precise in TrigonometricRegression (test 3)");
        assert_true(tr3.stats().r2 > 0.999, "The R2 should be bigger in TrigonometricRegression (test 3)");
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
        // First, we try to fill missings data by finding maxima or minima, wich is more precise but can fail
        initials_parameters_with_max_or_min(a, b, c, d);
        size_t n = this->m_data.size();
        // Then, we use more conventional methods
        if (!d)
        {
            d = this->sum_with_operation([](Coordinate<T> const& c) { return c.y(); }) / n;
        }
        if (!a) // a is supposed to be near the abs of the greatest y - the y mean
        {
            a = 0;
            for (size_t i = 0; i < n; i++)
            {
                T abs = std::abs(this->m_data[i].y() - d.value());
                if (abs > a)
                {
                    a = abs;
                }
            }
        }
        if (c.has_value())
        {
            c = mod(c.value(), std::numbers::pi_v<T> * 2); // C is defined %2*pi
        }
        return { std::vector<std::vector<T>> { { a.value() }, { b.value_or(1) }, { c.value_or(1) }, { d.value() } } };
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
            T x = this->m_data[i].x();
            r(i, 0) = std::sin(b * x + c);
            r(i, 1) = a * std::cos(b * x + c) + x;
            r(i, 2) = a * std::cos(b * x + c);
            r(i, 3) = 1;
        }
        return r;
    }
    virtual T predict_generic(Matrix<T> const& m, T const& v) const override
    {
        if (m.height() < 4)
        {
            throw Error(Error::Type::wrong_number_of_arguments_in_predict);
        }
        return m(0, 0) * std::sin(m(0, 1) * v + m(0, 2)) + m(0, 3);
    }
    void initials_parameters_with_max_or_min(std::optional<T>& a, std::optional<T>& b, std::optional<T>& c, std::optional<T>& d) const
    {
        if (a.has_value() && b.has_value() && c.has_value() && d.has_value())
        {
            return;
        }
        std::vector<Coordinate<T>> data = this->m_data;
        size_t n = data.size();
        std::sort(data.begin(), data.end(), [](Coordinate<T> const& c1, Coordinate<T> const& c2) { return c1.x() < c2.x(); });
        long int first = -1;
        bool first_is_maxima = false;
        long int second = -1;
        bool second_is_maxima = false;
        for (size_t i = 2; i < n - 2; i++)
        {
            // maxima
            if (data[i - 2].y() < data[i - 1].y() && data[i - 1].y() < data[i].y() && data[i].y() > data[i + 1].y() && data[i + 1].y() > data[i + 2].y())
            {
                if (first == -1)
                {
                    first = i;
                    first_is_maxima = true;
                }
                else
                {
                    second = i;
                    second_is_maxima = true;
                }
            }
            // minima
            else if (data[i - 2].y() > data[i - 1].y() && data[i - 1].y() > data[i].y() && data[i].y() < data[i + 1].y() && data[i + 1].y() < data[i + 2].y())
            {
                if (first == -1)
                {
                    first = i;
                }
                else
                {
                    second = i;
                }
            }
        }
        if (second == -1)
        {
            return;
        }
        if (!d)
        {
            if (first_is_maxima != second_is_maxima)
            {
                d = (data[first].y() + data[second].y()) / 2;
            }
            else
            {
                d = this->sum_with_operation([](Coordinate<T> const& c) { return c.y(); }) / n;
            }
        }
        if (!a)
        {
            T a1 = std::abs((first_is_maxima) ? data[first].y() - d.value() : d.value() - data[first].y());
            T a2 = std::abs((second_is_maxima) ? data[second].y() - d.value() : d.value() - data[second].y());
            a = (a1 + a2) / 2;
        }
        if (!b)
        {
            T r = data[second].x() - data[first].x();
            if (first_is_maxima != second_is_maxima)
            {
                r *= 2;
            }
            b = (2 * std::numbers::pi_v<T>) / r;
        }
        if (!c)
        {
            if (first_is_maxima)
            {
                c = (std::numbers::pi_v<T> / 2) - b.value() * data[first].x();
            }
            else if (second_is_maxima)
            {
                c = (std::numbers::pi_v<T> / 2) - b.value() * data[second].x();
            }
        }
    }
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
