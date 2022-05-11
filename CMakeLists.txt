cmake_minimum_required(VERSION 3.0.0)
project(drowsniess_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

if(MSVC)
  add_compile_options(/W3 /permissive-)
  string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus")
endif()

# Fetch from git
include(FetchContent)

FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG        4.4.0
        GIT_PROGRESS TRUE
)
FetchContent_GetProperties(opencv)
if (NOT opencv_POPULATED)
    FetchContent_Populate(opencv)
endif ()
FetchContent_MakeAvailable(opencv)


FetchContent_Declare(dlib
  GIT_REPOSITORY https://github.com/davisking/dlib.git
  GIT_TAG        v19.18
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(dlib)

add_executable(drowsniess_detection src/main.cpp)
target_link_libraries(drowsniess_detection PRIVATE dlib::dlib ${OpenCV_LIBS})
target_include_directories(drowsniess_detection PRIVATE ${OpenCV_INCLUDE_DIRS})