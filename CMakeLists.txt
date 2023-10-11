cmake_minimum_required(VERSION 3.0.0)
project(calibration VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)

include_directories(
    ./include
    ${OpenCV_INCLUDE_DIRS}
)
message("INFO ${OpenCV_INCLUDE_DIRS}")
message("INFO ${PROJECT_INCLUDE_DIR}")

add_executable(calibration main.cpp src/settings.cpp)
add_executable(test test.cpp)

target_include_directories(calibration PUBLIC ./include)

target_link_libraries(calibration ${OpenCV_LIBS})
target_link_libraries(test ${OpenCV_LIBS})