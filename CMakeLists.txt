cmake_minimum_required(VERSION 3.0)

project(matrix)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Debug")

set(SRCS
    errors.cpp
    main.cpp
)
    
set(HEADERS
    linear_regression.hpp
    regression.hpp
    concepts.hpp
    coordinate.hpp
    errors.hpp
    matrix.hpp
)

add_executable(matrix ${SRCS} ${HEADERS})
