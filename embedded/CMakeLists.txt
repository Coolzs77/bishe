# CMakeLists.txt
# RV1126嵌入式程序CMake配置

cmake_minimum_required(VERSION 3.10)
project(infrared_tracker VERSION 1.0.0 LANGUAGES CXX)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# 查找依赖库
find_package(OpenCV REQUIRED)

# RKNN运行时库路径
set(RKNN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/rknn/include")
set(RKNN_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/rknn/lib")

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIR}
)

# 链接目录
link_directories(
    ${RKNN_LIB_DIR}
)

# 源文件
set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/tracker.cpp
    src/pipeline.cpp
)

# 主程序
add_executable(infrared_tracker ${SOURCES})
target_link_libraries(infrared_tracker
    ${OpenCV_LIBS}
    rknnrt
    pthread
)

# 性能测试程序
add_executable(performance_test
    src/performance_test.cpp
    src/detector.cpp
)
target_link_libraries(performance_test
    ${OpenCV_LIBS}
    rknnrt
    pthread
)

# 安装
install(TARGETS infrared_tracker performance_test
    RUNTIME DESTINATION bin
)

install(DIRECTORY configs/
    DESTINATION configs
)
