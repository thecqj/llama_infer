# 在 Visual Studio 中禁用 CUDA 文件的自定义生成规则
if (MSVC)
    set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF CACHE BOOL "CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE")
endif ()

# 检查并启用 CUDA 支持
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.11)
    # 加载 CheckLanguage 模块，提供 check_language() 函数用于检测编程语言支持
    include(CheckLanguage)
    # 检测 CUDA 编译器是否存在
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER)    # 如果存在
        # 启用 CUDA 语言支持
        enable_language(CUDA)

        # 判断 CMake 版本是否 ≥ 3.17，决定使用现代还是旧版 CUDA 工具包查找方式
        if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)   # 使用现代 find_package(CUDAToolkit)
            find_package(CUDAToolkit QUIET)
            set(CUDA_TOOLKIT_INCLUDE ${CUDAToolkit_INCLUDE_DIRS})
        else () # 旧版 CMake 使用 find_package(CUDA)
            set(CUDA_FIND_QUIETLY TRUE)
            find_package(CUDA 11.0)
        endif ()

        # 标记 CUDA 已找到并记录版本
        set(CUDA_FOUND TRUE)
        set(CUDA_VERSION_STRING ${CMAKE_CUDA_COMPILER_VERSION})
    else ()
        # 未找到 CUDA 编译器时的提示
        message(STATUS "No CUDA compiler found")
    endif ()

else ()
    # 直接使用旧版 find_package(CUDA)，要求最低版本 11.0
    set(CUDA_FIND_QUIETLY TRUE)
    find_package(CUDA 11.0)
endif()

if (CUDA_FOUND)
    message(STATUS "Found CUDA Toolkit v${CUDA_VERSION_STRING}")

    # 包含自定义计算能力检测模块
    include(FindCUDA/select_compute_arch)

    # 检测系统中安装的 GPU 计算能力
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)

    # 清理并格式化 GPU 计算能力列表
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")

    # 设置 CMake CUDA 架构列表
    SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    MESSAGE(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

    # 确定 CUDA 库根目录
    if (DEFINED CMAKE_CUDA_COMPILER_LIBRARY_ROOT_FROM_NVVMIR_LIBRARY_DIR)
        set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "${CMAKE_CUDA_COMPILER_LIBRARY_ROOT_FROM_NVVMIR_LIBRARY_DIR}")
    elseif (EXISTS "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/nvvm/libdevice")
        set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")
    elseif (CMAKE_SYSROOT_LINK AND EXISTS "${CMAKE_SYSROOT_LINK}/usr/lib/cuda/nvvm/libdevice")
        set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "${CMAKE_SYSROOT_LINK}/usr/lib/cuda")
    elseif (EXISTS "${CMAKE_SYSROOT}/usr/lib/cuda/nvvm/libdevice")
        set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "${CMAKE_SYSROOT}/usr/lib/cuda")
    else ()
        message(FATAL_ERROR "Couldn't find CUDA library root.")
    endif ()
    # 清理临时变量
    unset(CMAKE_CUDA_COMPILER_LIBRARY_ROOT_FROM_NVVMIR_LIBRARY_DIR)
else ()
    # CUDA 未找到时的提示
    message(STATUS "CUDA was not found.")
endif ()