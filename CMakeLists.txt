cmake_minimum_required(VERSION 3.16)

# 生成 compile_commands.json 文件，用于代码补全工具（如 clangd）或静态分析
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 手动指定 CUDA 编译器路径
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")

# 启用 C++ 和 CUDA 语言支持，项目名称为 llama_infer
project(llama_infer CXX CUDA)

# 包含 CUDA 配置
include(cmake/cuda.cmake)

# 必须支持C++17和CUDA的C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# 功能选项定义
option(LLAMA3_SUPPORT OFF)  # 默认关闭“支持llama3”选项
if (LLAMA3_SUPPORT)
    message(STATUS "LLAMA3 SUPPORT")
    add_definitions(-DLLAMA3_SUPPORT)
endif()

option(QWEN2_SUPPORT OFF)   # 默认关闭“支持qwen2”选项
if (QWEN2_SUPPORT)
    message(STATUS "QWEN2 SUPPORT")
    add_definitions(-DQWEN2_SUPPORT)
endif()

# 依赖管理（CPM）
option(USE_CPM "Use CPM for dependency management" OFF)

if(USE_CPM)
    include(cmake/CPM.cmake)

    CPMAddPackage(
        NAME GTest
        GITHUB_REPOSITORY google/googletest
        VERSION 1.15.0
    )

    CPMAddPackage(
        NAME glog
        GITHUB_REPOSITORY google/glog
        VERSION 0.7.1
        OPTIONS "BUILD_TESTING Off"
    )

    CPMAddPackage(
        NAME Armadillo
        GITLAB_REPOSITORY conradsnicta/armadillo-code
        GIT_TAG 14.0.1
    )

    CPMAddPackage(
        NAME sentencepiece
        GITHUB_REPOSITORY google/sentencepiece
        VERSION 0.2.0
    )
    find_package(sentencepiece REQUIRED)

    if (LLAMA3_SUPPORT OR QWEN2_SUPPORT)
        CPMAddPackage(
            NAME absl
            GITHUB_REPOSITORY abseil/abseil-cpp
            GIT_TAG 20240722.0
            OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
        )
        CPMAddPackage(
            NAME re2
            GITHUB_REPOSITORY google/re2
            GIT_TAG 2024-07-02
        )
        CPMAddPackage(
            NAME nlohmann_json
            GITHUB_REPOSITORY nlohmann/json
            VERSION 3.11.3
        )
    endif()
endif()

# ---------------------------------
# 依赖查找
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

# 源文件收集
aux_source_directory(kuiper/source/base/ DIR_BASE)
aux_source_directory(kuiper/source/tensor/ DIR_TENSOR)

# 动态库构建
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# 添加源文件路径
add_library(llama SHARED ${DIR_BASE} ${DIR_TENSOR})
# 需要链接的库和依赖项
target_link_libraries(llama sentencepiece glog::glog gtest gtest_main pthread cudart armadillo)
# 添加 CUDA 库的路径
target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

# 添加头文件搜索路径
target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/kuiper/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(USE_CPM)
    # Add sentencepiece include directory
    target_include_directories(llama PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()

# 启用 CUDA 可分离编译
set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 子目录添加（目录下有 CMakeLists.txt）
add_subdirectory(test)
# add_subdirectory(demo)
