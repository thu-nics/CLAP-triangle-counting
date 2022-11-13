CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(CAM_architecture_simulation)
set(CMAKE_BUILD_TYPE "release")

add_executable(CAM_architecture_simulation ./src/main.cpp ./src/mem.cpp)