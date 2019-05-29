cmake_minimum_required(VERSION 3.5)
project(TinyInfer)

set(CMAKE_CXX_STANDARD 14)

# For protobuf
set(PROTOC /usr/local/bin/protoc)
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

# =========================== Protobuf (On Cmake) ===========================
set(Protobuf_LIBRARIES /usr/local/lib/libprotobuf.so)
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# IDE would be happy to have proto files compiled on cmake
file(GLOB PROTO_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} proto/*.proto)
execute_process(
        #        COMMAND ${PROTOC} -I=proto --cpp_out=proto_built ${PROTO_SRC}
        COMMAND ./scripts/compile_pb.sh ${PROTOC} ${CMAKE_CURRENT_SOURCE_DIR}/src/proto ${CMAKE_CURRENT_SOURCE_DIR}/src/proto_built ${CMAKE_CURRENT_BINARY_DIR}/src/proto_built
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)




# ======================= Protobuf (Delta Compilation) ==========================
# set up protobuf compilation command
ADD_CUSTOM_COMMAND(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/proto_cache_file
        COMMAND ./scripts/compile_pb.sh ${PROTOC} ${CMAKE_CURRENT_SOURCE_DIR}/src/proto ${CMAKE_CURRENT_SOURCE_DIR}/src/proto_built ${CMAKE_CURRENT_BINARY_DIR}/src/proto_built
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
        proto_cache_target ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/proto_cache_file
)

# ========================== For testing execuables ==============================
include(${PROJECT_SOURCE_DIR}/cmake/helper.cmake)
SUBDIRLIST(LINKER_INLCUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src ./src)
include_directories(${LINKER_INLCUDE_DIRS})
include_directories(src)

add_subdirectory(src)

add_executable(main test/main.cpp)
target_link_libraries(main TinyinferLib)

add_executable(parser_test test/parserTest.cpp)
target_link_libraries(parser_test TinyinferLib)

# and finally add dependency of the main target
ADD_DEPENDENCIES(main proto_cache_target)

# unit test
file(GLOB UTIL_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/infer/utils/*)
include_directories(${PROJECT_SOURCE_DIR}/infer/utils)

add_executable( test_relu1 ${PROJECT_SOURCE_DIR}/test/test_relu1.cpp ${UTIL_SRC})
target_link_libraries( test_relu1 Eigen3::Eigen )

add_executable( test_relu2 ${PROJECT_SOURCE_DIR}/test/test_relu2.cpp ${UTIL_SRC})
target_link_libraries( test_relu2 Eigen3::Eigen )

add_executable( test_dense1 ${PROJECT_SOURCE_DIR}/test/test_dense1.cpp ${UTIL_SRC})
target_link_libraries( test_dense1 Eigen3::Eigen )

add_executable( test_dense2 ${PROJECT_SOURCE_DIR}/test/test_dense2.cpp ${UTIL_SRC})
target_link_libraries( test_dense2 Eigen3::Eigen )

add_executable( test_dense3 ${PROJECT_SOURCE_DIR}/test/test_dense3.cpp ${UTIL_SRC})
target_link_libraries( test_dense3 Eigen3::Eigen )

add_executable( test_flatten1 ${PROJECT_SOURCE_DIR}/test/test_flatten1.cpp ${UTIL_SRC})
target_link_libraries( test_flatten1 Eigen3::Eigen )

add_executable( test_maxpool1 ${PROJECT_SOURCE_DIR}/test/test_maxpool1.cpp ${UTIL_SRC})
target_link_libraries( test_maxpool1 Eigen3::Eigen )

add_executable( test_conv1 ${PROJECT_SOURCE_DIR}/test/test_conv1.cpp ${UTIL_SRC})
target_link_libraries( test_conv1 Eigen3::Eigen )

add_executable( test_conv2 ${PROJECT_SOURCE_DIR}/test/test_conv2.cpp ${UTIL_SRC})
target_link_libraries( test_conv2 Eigen3::Eigen )


