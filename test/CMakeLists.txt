# 包含自定义 CUDA 配置
include(../cmake/cuda.cmake)

# 查找依赖包
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

# 定义变量 link_ext_lib，包含要链接的外部库目标
set(link_ext_lib glog::glog GTest::gtest)

# 收集测试源文件
aux_source_directory(../test DIR_TEST)
aux_source_directory(../test/buffer DIR_TEST_BUFFER)
aux_source_directory(../test/tensor DIR_TEST_TENSOR)
aux_source_directory(../test/op DIR_TEST_OP)

# 创建可执行目标
add_executable(test_llm ${DIR_TEST} ${DIR_TEST_BUFFER} ${DIR_TEST_TENSOR} ${DIR_TEST_OP})

# 将 glog 和 GTest 链接到 test_llm 可执行文件
target_link_libraries(test_llm ${link_ext_lib})

# 添加头文件目录
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ../kuiper/include)

# 添加链接目录
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)

# 条件链接额外依赖
if (LLAMA3_SUPPORT OR QWEN2_SUPPORT)
    message(STATUS "LINK LLAMA3 SUPPORT")
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(llama absl::base re2::re2 nlohmann_json::nlohmann_json)
endif ()

# 将项目内的 llama 库链接到 test_llm 可执行文件
target_link_libraries(test_llm llama)

# 设置目标属性
set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)