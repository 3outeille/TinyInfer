cmake_minimum_required(VERSION 3.5)
project(TinyInfer)

set(CMAKE_CXX_STANDARD 14)



# =============================== Eigen 3 ==================================
set(Eigen3_DIR ${CMAKE_CURRENT_BINARY_DIR}/eigen-build/share/eigen3/cmake/)

include(${PROJECT_SOURCE_DIR}/cmake/misc.cmake)
include_directories(${PROJECT_SOURCE_DIR}/include)

configure_file(${PROJECT_SOURCE_DIR}/cmake/get_eigen.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/eigen-download/CMakeLists.txt)
setup_eigen()



