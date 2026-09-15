cmake_minimum_required(VERSION 3.24)
project(FlyVoxelisedReality VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc video videoio)

add_executable(FlyVoxelisedReality
    src/main.cpp
)

target_link_libraries(FlyVoxelisedReality PRIVATE
    OpenGL::GL
    glad::glad
    glfw
    glm::glm
    ${OpenCV_LIBS}
)

target_include_directories(FlyVoxelisedReality PRIVATE ${OpenCV_INCLUDE_DIRS})

if(MSVC)
    target_compile_options(FlyVoxelisedReality PRIVATE /W4 /permissive-)
else()
    target_compile_options(FlyVoxelisedReality PRIVATE -Wall -Wextra -Wpedantic)
endif()
