# 版权声明
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.5.1)

# 定义项目名称
project(Sample_YOLO11-OrangepiAIPro)

# 添加编译选项，指定 C++11 标准
add_compile_options(-std=c++11)

# 添加宏定义
add_definitions(-DENABLE_DVPP_INTERFACE)

# 设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../out")

# 设置调试模式下的编译器标志
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")

# 设置发布模式下的编译器标志
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 设置 INC_PATH 环境变量
set(INC_PATH $ENV{DDK_PATH})

# 检查 DDK_PATH 环境变量是否已定义
if (NOT DEFINED ENV{DDK_PATH})
    # 如果未定义，则设置默认路径
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    # 如果已定义，则使用环境变量中的路径
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()

# 设置 LIB_PATH 环境变量
set(LIB_PATH $ENV{NPU_HOST_LIB})

# 检查 NPU_HOST_LIB 环境变量是否已定义
if (NOT DEFINED ENV{NPU_HOST_LIB})
    # 如果未定义，则设置默认路径
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/lastest/runtime/lib64/stub")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    # 如果已定义，则使用环境变量中的路径
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()

# 设置 THIRDPART 环境变量
set(THIRDPART $ENV{THIRDPART_PATH})

# 检查 THIRDPART_PATH 环境变量是否已定义
if (NOT DEFINED ENV{THIRDPART_PATH})
    # 如果未定义，则设置默认路径
    set(THIRDPART "/usr/local/Ascend/ascend-toolkit/latest/thirdpart")
    message(STATUS "set default THIRDPART: ${THIRDPART}")
else()
    # 如果已定义，则使用环境变量中的路径
    message(STATUS "set THIRDPART: ${THIRDPART}")
endif()

# 包含指定目录
include_directories(
   ${INC_PATH}/runtime/include/
   ${THIRDPART}/include/acllite
    /usr/include/opencv4/
)

# 链接指定目录
link_directories(
    ${THIRDPART}/lib/
    ${LIB_PATH}
)

# 添加可执行文件目标
add_executable(main
        Sample_YOLO11-OrangepiAIPro.cpp
)

# 根据目标类型链接不同库
if(target STREQUAL "Simulator_Function")
    target_link_libraries(main funcsim)
else()
    target_link_libraries(main ascendcl acl_dvpp stdc++ acllite dl rt opencv_core opencv_imgproc opencv_imgcodecs)
endif()

# 安装目标文件到指定目录
install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
