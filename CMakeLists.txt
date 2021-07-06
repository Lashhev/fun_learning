cmake_minimum_required(VERSION 3.10)

project(fun_learning VERSION 0.1.1)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
set(CMAKE_CXX_STANDARD 17) # use C++14
if(CMAKE_BUILD_TYPE EQUAL RELEASE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -funsafe-math-optimizations -O3 -pipe")
endif()

option(BUILD_FUN_LEARNING_EXAMPLE "Build fun learning example" ON)

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp 0.5.2 REQUIRED)

string(TIMESTAMP TODAY "%Y-%m-%d")
configure_file(config/version_info.h.in version_info.h)
# include_directories(include ${YAML_CPP_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(activation_functions src/activation_functions.cpp)
target_link_libraries(activation_functions Eigen3::Eigen)
target_include_directories(activation_functions PUBLIC include)

add_library(neural_layer src/neural_layer.cpp)
target_link_libraries(neural_layer activation_functions ${YAML_CPP_LIBRARIES})
target_include_directories(neural_layer PUBLIC ${YAML_CPP_INCLUDE_DIR})

add_library(${PROJECT_NAME} src/neural_network.cpp)
target_link_libraries(${PROJECT_NAME} neural_layer)

if(BUILD_FUN_LEARNING_EXAMPLE)
    add_subdirectory(example)
endif()


