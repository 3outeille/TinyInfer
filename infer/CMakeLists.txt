cmake_minimum_required(VERSION 3.5)
project(loader)

set(CMAKE_CXX_STANDARD 14)
set(PYBIND11_CPP_STANDARD -std=c++14)
set(PROTOC /usr/local/bin/protoc)

# For protobuf
set(CMAKE_CXX_FLAGS "-pthread -I/usr/local/include -L/usr/local/lib -lprotobuf -pthread")


# =============================== JSON =====================================
find_package(nlohmann_json 3.2.0 REQUIRED)

# =============================== Eigen 3 ==================================
set(Eigen3_DIR ${CMAKE_CURRENT_BINARY_DIR}/eigen-build/share/eigen3/cmake/)

include(${PROJECT_SOURCE_DIR}/cmake/misc.cmake)
include_directories(${PROJECT_SOURCE_DIR}/include)

configure_file(${PROJECT_SOURCE_DIR}/cmake/get_eigen.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/eigen-download/CMakeLists.txt)
setup_eigen()

# find_package(pybind11)
#find_package (Eigen3 REQUIRED NO_MODULE)
#add_subdirectory(pybind11)


# =========================== Protobuf (On Cmake) ===========================
set(Protobuf_LIBRARIES /usr/local/lib/libprotobuf.so)
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# IDE would be happy to have proto files compiled on cmake
file(GLOB PROTO_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} proto/*.proto)
execute_process(
        COMMAND mkdir -p proto_built
        COMMAND ${PROTOC} -I=proto --cpp_out=proto_built ${PROTO_SRC}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)


#MESSAGE("PB Start")

#MESSAGE("files: ${PROTO_SRC}")

#execute_process(
#        COMMAND protoc ${PROTO_SRC} -I=proto --cpp_out=proto_built
#        COMMAND pwd
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#)
#MESSAGE("PB Done")


# ============================= Executable ===================================
link_directories(proto_built)

file(GLOB PROTO_BUILT_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} proto_built/*.h)
file(GLOB PROTO_BUILT_SRC ${CMAKE_CURRENT_SOURCE_DIR} proto_built/*.cc)
add_executable(main main.cpp ${PROTO_BUILT_INCLUDE} ${PROTO_BUILT_SRC})
target_include_directories(main PUBLIC proto_built)
target_link_libraries(main nlohmann_json::nlohmann_json Eigen3::Eigen ${Protobuf_LIBRARIES})
#target_link_libraries(main Eigen3::Eigen)
#target_link_libraries(load_tf PRIVATE Eigen3::Eigen pybind11::embed)

# ======================= Protobuf (On Compilation) ==========================
# set up protobuf compilation command
ADD_CUSTOM_COMMAND(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/proto_cache_file
        COMMAND mkdir -p proto_built
        COMMAND ${PROTOC} -I=proto --cpp_out=proto_built ${PROTO_SRC}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
        proto_cache_target ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/proto_cache_file
)

# and finally add dependency of the main target
ADD_DEPENDENCIES(main proto_cache_target)

