# ============================= Executable ==================================
include(${PROJECT_SOURCE_DIR}/cmake/helper.cmake)

file(GLOB_RECURSE SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.cpp)
#list(REMOVE_ITEM SRC_FILES "src/main.cpp")
file(GLOB_RECURSE HPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.hpp)

file(GLOB_RECURSE PROTO_HEAD_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.pb.h)
file(GLOB_RECURSE PROTO_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.pb.cc)

#message("SRC_FILES: ${SRC_FILES}")

SUBDIRLIST(LINKER_INLCUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src .)
#MESSAGE("LINKER_INLCUDE_DIRS: ${LINKER_INLCUDE_DIRS}")
#include_directories(${LINKER_INLCUDE_DIRS})

#message("PROTO_HEAD_FILES: ${PROTO_HEAD_FILES}")

include_directories(${LINKER_INLCUDE_DIRS})
include_directories(".")

#add_executable(main src/main.cpp ${PROTO_HEAD_FILES} ${PROTO_SRC_FILES} ${HPP_FILES} ${SRC_FILES})
add_library(TinyinferLib STATIC ${SRC_FILES} ${HPP_FILES} ${PROTO_HEAD_FILES} ${PROTO_SRC_FILES})
#target_include_directories(main PUBLIC src src/proto_built)
target_link_libraries(TinyinferLib nlohmann_json::nlohmann_json Eigen3::Eigen ${Protobuf_LIBRARIES})