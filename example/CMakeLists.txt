cmake_minimum_required(VERSION 3.10)
project(fun_learning_example)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries (${PROJECT_NAME} fun_learning)