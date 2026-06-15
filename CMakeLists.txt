# 1. 声明 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 2. 项目名称和语言 (因为你有 .cpp 文件，所以包含 C 和 CXX)
project(lubancat_gst_core LANGUAGES C CXX)

# 3. 设置 C/C++ 标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

# 4. 查找依赖库 (GStreamer)
# 这一步会自动去找系统中 GStreamer 的头文件和库路径
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-video-1.0 gstreamer-allocators-1.0 gstreamer-app-1.0)

# 5. 收集所有的源文件 (.c 和 .cpp)
# 使用 aux_source_directory 或者手动指定，手动指定更清晰
set(SOURCE_FILES
    src/camera/v4l2_camera.c
    src/memory/dma_buffer.c
    test/test.cpp
    src/app/vision_pipeline.cpp
    src/inference/rknn_model.cpp
    src/inference/postprocess.cc
    src/rga/gst_rga.c
    tool/draw_lines_by_cpu.cpp
    # 如果以后增加了新的模块，直接在这里添加 .c 或 .cpp 文件即可
)

# 6. 定义可执行文件目标
add_executable(gst_app ${SOURCE_FILES})

# 显式指定使用 C++ 链接器，确保链接 libstdc++
set_target_properties(gst_app PROPERTIES LINKER_LANGUAGE CXX)

# 7. 解决“找不到头文件”的问题：设置头文件搜索路径
# 这里把所有子目录都加进去，你在代码里就可以直接 #include "dma_buffer.h"
target_include_directories(gst_app PRIVATE
    src/camera
    src/memory
    src/app
    src/inference
    src/rga
    tool/
    common
    include/
    ${GSTREAMER_INCLUDE_DIRS}  # GStreamer 的头文件路径
)

# 8. 解决“undefined reference”的问题：链接库文件
# 链接 GStreamer 的库，以及 Linux 下常用的 pthread 和数学库
target_link_libraries(gst_app PRIVATE
    ${GSTREAMER_LIBRARIES}
    pthread
    rga
    /home/cat/rknn_yolov5_demo/lib/librknnrt.so
)