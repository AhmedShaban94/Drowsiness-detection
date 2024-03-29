cmake_minimum_required(VERSION 3.0.0)
project(drowsniess_detection)

find_package(dlib CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

if(MSVC)
  add_compile_options(/W3 /permissive-)
  string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus")
endif()


add_executable(drowsniess_detection src/main.cpp)
target_link_libraries(drowsniess_detection PRIVATE dlib::dlib ${OpenCV_LIBS})
target_include_directories(drowsniess_detection PRIVATE ${OpenCV_INCLUDE_DIRS})

add_custom_command(
                    TARGET drowsniess_detection PRE_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_SOURCE_DIR}/utils $<TARGET_FILE_DIR:drowsniess_detection>/utils)