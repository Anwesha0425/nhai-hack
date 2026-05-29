cmake_minimum_required(VERSION 3.18)
project(face_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fPIC")

find_library(LOG_LIB log)

set(TFLITE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../../libs/tflite/include")
set(TFLITE_LIB_DIR "${CMAKE_SOURCE_DIR}/../../../libs/tflite/${ANDROID_ABI}")

message(STATUS "Using prebuilt TFLite from libs/tflite")
add_library(tensorflowlite SHARED IMPORTED)
set_target_properties(tensorflowlite PROPERTIES
    IMPORTED_LOCATION "${TFLITE_LIB_DIR}/libtensorflowlite_jni.so"
)
set(TFLITE_HEADERS ${TFLITE_INCLUDE_DIR})

add_library(face_pipeline SHARED
    face_pipeline.cpp
)

target_include_directories(face_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${TFLITE_HEADERS}
)

target_link_libraries(face_pipeline
    ${LOG_LIB}
    tensorflowlite
)
