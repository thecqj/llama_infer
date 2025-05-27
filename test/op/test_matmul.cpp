#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>

#include "op/matmul.h"

TEST(test_matmul, test_cpu) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    const int32_t dim1 = 3;
    const int32_t dim0 = 2;

    // [7, 8, 9]
    Tensor input(DataType::kDataTypeFp32, dim1, true, alloc);
    input.index<float>(0) = 7;
    input.index<float>(1) = 8;
    input.index<float>(2) = 9;

    // [1, 2, 3
    //  4, 5, 6]
    Tensor weight(DataType::kDataTypeFp32, dim0, dim1, true, alloc);
    for (int i = 1; i <= dim0 * dim1; ++i) {
        weight.index<float>(i - 1) = i;
    }

    // it should be [50, 122]
    Tensor output(DataType::kDataTypeFp32, dim0, true, alloc);

    MatmulLayer matmul_layer(DeviceType::kDeviceCPU, DataType::kDataTypeFp32, dim0, dim1);
    LayerParam& layer = matmul_layer;
    layer.set_weight(0, weight);
    layer.forward(input, output);

    ASSERT_EQ(input.size(), dim1);
    ASSERT_EQ(input.dims_size(), 1);
    ASSERT_EQ(weight.size(), dim0 * dim1);
    ASSERT_EQ(weight.dims_size(), 2);
    ASSERT_EQ(output.size(), dim0);
    ASSERT_EQ(output.dims_size(), 1);

    ASSERT_NEAR(output.index<float>(0), 50, 0.01);
    ASSERT_NEAR(output.index<float>(1), 122, 0.01);
}

TEST(test_matmul, test_gpu_fp32) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    const int32_t dim0 = 256;
    const int32_t dim1 = 2048;

    std::random_device rd;  // 使用硬件熵源生成真随机数种子
    std::mt19937 mt(rd());  // 用种子初始化 Mersenne Twister 伪随机数引擎
    std::uniform_real_distribution<float> dist(0.f, 1.f);   // 定义均匀分布，范围 [0.0, 1.0)

    Tensor input_cpu(DataType::kDataTypeFp32, dim1, true, alloc);
    for (int i = 0; i < dim1; ++i) {
        input_cpu.index<float>(i) = dist(mt);
    }
    Tensor weight_cpu(DataType::kDataTypeFp32, dim0, dim1, true, alloc);
    for (int i = 0; i < dim0 * dim1; ++i) {
        weight_cpu.index<float>(i) = dist(mt);
    }
    Tensor output_cpu(DataType::kDataTypeFp32, dim0, true, alloc);

    // GPU 计算
    Tensor input_gpu = input_cpu.clone();
    Tensor weight_gpu = weight_cpu.clone();
    Tensor output_gpu = output_cpu.clone();
    input_gpu.to_cuda();
    weight_gpu.to_cuda();
    output_gpu.to_cuda();

    MatmulLayer matmul_layer_gpu(DeviceType::kDeviceGPU, DataType::kDataTypeFp32, dim0, dim1);
    LayerParam& layer_gpu = matmul_layer_gpu;
    layer_gpu.set_cuda_config(std::make_shared<kernel::CudaConfig>());
    layer_gpu.set_weight(0, weight_gpu);
    layer_gpu.forward(input_gpu, output_gpu);

    output_gpu.to_cpu();

    // CPU 计算
    MatmulLayer matmul_layer_cpu(DeviceType::kDeviceCPU, DataType::kDataTypeFp32, dim0, dim1);
    LayerParam& layer_cpu = matmul_layer_cpu;
    layer_cpu.set_weight(0, weight_cpu);
    layer_cpu.forward(input_cpu, output_cpu);

    // 验证
    for (int i = 0; i < output_cpu.size(); ++i) {
        ASSERT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 1e-2);
    }
}