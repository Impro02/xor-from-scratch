cmake_minimum_required(VERSION 3.10)

project(HelloWorld)

find_package(fmt CONFIG REQUIRED)
find_package(eigen3 CONFIG REQUIRED)


add_executable(HelloWorld 
    src/main.cpp
    src/network/network.cpp
    src/functions/activation/tanh_function.cpp
    src/functions/loss/loss.cpp
    src/layer/activation_layer.cpp
    src/layer/fc_layer.cpp
)

include_directories(${CMAKE_SOURCE_DIR}/src)

target_link_libraries(HelloWorld PRIVATE fmt::fmt Eigen3::Eigen)