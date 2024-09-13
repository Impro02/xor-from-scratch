cmake_minimum_required(VERSION 3.10)

project(XorFromScratch)

find_package(fmt CONFIG REQUIRED)
find_package(eigen3 CONFIG REQUIRED)


add_executable(XorFromScratch 
    src/main.cpp
    src/network/network.cpp
    src/functions/activation/tanh_function.cpp
    src/functions/loss/mse_loss_function.cpp
    src/layers/activation_layer.cpp
    src/layers/fc_layer.cpp
)

include_directories(${CMAKE_SOURCE_DIR}/include)

target_link_libraries(XorFromScratch PRIVATE fmt::fmt Eigen3::Eigen)