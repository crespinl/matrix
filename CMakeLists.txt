cmake_minimum_required(VERSION 3.0)

project(matrix)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Debug")

set(SRCS
    errors.cpp
)
    
set(HEADERS
    lm_regression.hpp
    trigonometric_regression.hpp
    fourier.hpp
    regression_stats.hpp
    power_regression.hpp
    exponential_regression_2.hpp
    logarithmic_regression.hpp
    exponential_regression.hpp
    polynomial_regression.hpp
    proportional_regression.hpp
    tests.hpp
    linear_regression.hpp
    regression.hpp
    concepts.hpp
    coordinate.hpp
    errors.hpp
    matrix.hpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fdiagnostics-color=always")
add_executable(matrix ${SRCS} main.cpp ${HEADERS})

project(matrix_lib)

add_library(matrix_lib STATIC ${SRCS} ${HEADERS})