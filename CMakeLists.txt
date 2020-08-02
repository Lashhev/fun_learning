cmake_minimum_required(VERSION 3.10)

project(fun_learning)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
set(CMAKE_CXX_STANDARD 17) # use C++14
set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 REQUIRED)

include_directories(include)

add_library(activation_functions src/activation_functions.cpp)
target_link_libraries(activation_functions Eigen3::Eigen)

add_library(neural_layer src/neural_layer.cpp)
target_link_libraries(neural_layer activation_functions)

add_library(neural_network src/neural_network.cpp)
target_link_libraries(neural_network neural_layer)

add_executable(${PROJECT_NAME}_test main.cpp)
target_link_libraries (${PROJECT_NAME}_test neural_network)
