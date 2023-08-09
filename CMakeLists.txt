cmake_minimum_required(VERSION 3.9)

project(matrix)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Release")

set(SRCS
    errors.cpp
)
    
set(HEADERS
    gaussian_regression.hpp
    logistic_regression.hpp
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
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -Og")
set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS} -O3 -march=native -mfpmath=sse")
add_executable(matrix ${SRCS} main.cpp ${HEADERS})

project(matrix_lib)

add_library(matrix_lib STATIC ${SRCS} ${HEADERS})

find_package(OpenMP)
if(OpenMP_FOUND)
    target_link_libraries(matrix PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(matrix_lib PUBLIC OpenMP::OpenMP_CXX)
    add_compile_definitions(USE_OPENMP)
else()
    message("Warning : compiling without OpenMP will decrease performance")
endif()