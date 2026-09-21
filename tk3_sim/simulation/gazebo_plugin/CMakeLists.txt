cmake_minimum_required(VERSION 3.10)
project(constant_wrench_system)

find_package(gz-sim9 REQUIRED)
find_package(gz-plugin3 REQUIRED)
find_package(gz-transport14 REQUIRED)

add_library(constant_wrench_system SHARED constant_wrench_system.cc)
target_compile_features(constant_wrench_system PRIVATE cxx_std_17)
target_link_libraries(constant_wrench_system PRIVATE
  gz-sim9::gz-sim9
  gz-plugin3::gz-plugin3
  gz-transport14::gz-transport14
)

install(TARGETS constant_wrench_system LIBRARY DESTINATION lib/gazebo)
