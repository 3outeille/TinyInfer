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


# ============================= Executable ==================================
include(${PROJECT_SOURCE_DIR}/cmake/helper.cmake)

file(GLOB_RECURSE SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp)
list(REMOVE_ITEM SRC_FILES "src/main.cpp")
file(GLOB_RECURSE HPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.hpp)

file(GLOB_RECURSE PROTO_HEAD_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.pb.h)
file(GLOB_RECURSE PROTO_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.pb.cc)

#message("SRC_FILES: ${SRC_FILES}")

SUBDIRLIST(LINKER_INLCUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src src)
#MESSAGE("LINKER_INLCUDE_DIRS: ${LINKER_INLCUDE_DIRS}")
#include_directories(${LINKER_INLCUDE_DIRS})

#message("PROTO_HEAD_FILES: ${PROTO_HEAD_FILES}")

include_directories(${LINKER_INLCUDE_DIRS})

add_executable(main src/main.cpp ${PROTO_HEAD_FILES} ${PROTO_SRC_FILES} ${HPP_FILES} ${SRC_FILES})
#target_include_directories(main PUBLIC src src/proto_built)
target_link_libraries(main nlohmann_json::nlohmann_json Eigen3::Eigen ${Protobuf_LIBRARIES})

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

# and finally add dependency of the main target
ADD_DEPENDENCIES(main proto_cache_target)


