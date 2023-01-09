cmake_minimum_required(VERSION 3.0)

project(matrix)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Debug")

set(SRCS
    errors.cpp
    main.cpp
)
    
set(HEADERS
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

add_executable(matrix ${SRCS} ${HEADERS})
