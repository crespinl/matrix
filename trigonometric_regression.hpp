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
    {
        std::sort(this->m_data.begin(), this->m_data.end(), [](Coordinate<T> const& c1, Coordinate<T> const& c2) { return c1.x() < c2.x(); });
        // Points need to be sorted here for the approximation algorithm
    }

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
    static void fuzzer()
    {
        long double a = 1;
        long double b = 0.3;
        long double c = 3;
        long double d = 4;
        int const n = 50;
        long double const min = 0;
        long double const max = 20;
        std::vector<Coordinate<long double>> points;
        points.reserve(n);
        std::default_random_engine re { (long unsigned int) std::chrono::system_clock::now().time_since_epoch().count() };
        std::uniform_real_distribution<long double> generator(min, max);
        auto func = [a, b, c, d](long double v) { return a * std::sin(b * v + c) + d; };
        for (size_t i = 0; i < n; i++)
        {
            long double v = generator(re);
            points.push_back({ v, func(v) });
        }
        TrigonometricRegression<long double> regression { points };
        regression.calculate_model();
        std::cout << "a : used " << a << " and calculated " << regression.a() << std::endl;
        std::cout << "b : used " << b << " and calculated " << regression.b() << std::endl;
        std::cout << "c : used " << c << " and calculated " << regression.c() << std::endl;
        std::cout << "d : used " << d << " and calculated " << regression.d() << std::endl;
        std::cout << "Got a R2 of " << regression.stats().r2 << std::endl;
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
        if (!initials_parameters_with_integral_equation(a, b, c, d))
        {
            // If it didn't work, we use easier methods
            initials_parameters_with_max_or_min(a, b, c, d);
            size_t n = this->m_data.size();
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

    bool initials_parameters_with_integral_equation(std::optional<T>& a, std::optional<T>& b, std::optional<T>& c, std::optional<T>& d) const
    /*
    This function computes an approximation for the parameters using an algorithm based on integral equations.
    In does some failible operations, such as matrix inversions, wich means that in can sometimes return false if failed.
    */
    {
        if (a && b && c && d)
        {
            return true;
        }
        size_t const n = this->m_data.size();
        if (n < 5)
        {
            // If the number of parameters is too small, the precision becomes verry bad, so we prefer to use other methods
            return false;
        }
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
        T square_x_sum = this->sum_with_operation([](Coordinate<T> const& c) { return std::pow<T>(c.x(), 2); });
        T power_three_x_sum = this->sum_with_operation([](Coordinate<T> const& c) { return std::pow<T>(c.x(), 3); });
        T power_four_x_sum = this->sum_with_operation([](Coordinate<T> const& c) { return std::pow<T>(c.x(), 4); });
        T SS_sum = 0;
        for (auto const& e : SS)
        {
            SS_sum += e;
        }
        T square_SS_sum = 0;
        for (auto const& e : SS)
        {
            square_SS_sum += std::pow<T>(e, 2);
        }
        T x_times_SS_sum = 0;
        T y_times_SS_sum = 0;
        T x_square_times_SS_sum = 0;
        for (size_t i = 0; i < n; i++)
        {
            x_times_SS_sum += this->m_data[i].x() * SS[i];
            y_times_SS_sum += this->m_data[i].y() * SS[i];
            x_square_times_SS_sum += std::pow<T>(this->m_data[i].x(), 2) * SS[i];
        }
        T y_sum = this->sum_with_operation([](Coordinate<T> const& c) { return c.y(); });
        T y_times_x_sum = this->sum_with_operation([](Coordinate<T> const& c) { return c.y() * c.x(); });
        T y_times_square_x_sum = this->sum_with_operation([](Coordinate<T> const& c) { return c.y() * std::pow<T>(c.x(), 2); });

        Matrix<T> m1 { {
            { square_SS_sum, x_square_times_SS_sum, x_times_SS_sum, SS_sum },
            { x_square_times_SS_sum, power_four_x_sum, power_three_x_sum, square_x_sum },
            { x_times_SS_sum, power_three_x_sum, square_x_sum, x_sum },
            { SS_sum, square_x_sum, x_sum, static_cast<T>(n) },
        } };
        Matrix<T> m2 { {
            { y_times_SS_sum },
            { y_times_square_x_sum },
            { y_times_x_sum },
            { y_sum },
        } };
        try
        {
            m1.inverse();
        }
        catch (...)
        {
            return false;
        }
        // TODO : check
        auto result_one = m1 * m2;
        T omega_1 = std::sqrt(-result_one(0, 0));
        T a_1 = 2. * result_one(0, 1) / std::pow(omega_1, 2);
        T x_1 = this->m_data[0].x();
        T big_calculus_1 = result_one(0, 1) * std::pow(x_1, 2) + result_one(0, 2) * x_1 + result_one(0, 3) - a_1;
        T big_calculus_2 = (result_one(0, 2) + 2 * result_one(0, 1) * x_1) / omega_1;
        T big_calculus_3 = omega_1 * x_1;
        T b_1 = big_calculus_1 * std::sin(big_calculus_3) + big_calculus_2 * std::cos(big_calculus_3);
        T c_1 = big_calculus_1 * std::cos(big_calculus_3) - big_calculus_2 * std::sin(big_calculus_3);

        T rho_1 = std::sqrt(std::pow(b_1, 2) + std::pow(c_1, 2));
        T phi_1 = std::acos(b_1 / rho_1);

        if (!a)
            a = rho_1;
        if (!b)
            b = omega_1;
        if (!c)
            c = phi_1;
        if (!d)
            d = a_1;
        // Not use the end of the algorithm for now
        return true;
    }

    void initials_parameters_with_max_or_min(std::optional<T>& a, std::optional<T>& b, std::optional<T>& c, std::optional<T>& d) const
    {
        if (a.has_value() && b.has_value() && c.has_value() && d.has_value())
        {
            return;
        }
        std::vector<Coordinate<T>> const& data = this->m_data;
        size_t n = data.size();
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
