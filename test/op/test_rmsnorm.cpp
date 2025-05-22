#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>

#include "op/rmsnorm.h"

TEST(test_rmsnorm, test_cpu) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    const int32_t size = 128;
    Tensor input(DataType::kDataTypeFp32, size, true, alloc);
    Tensor weight(DataType::kDataTypeFp32, size, true, alloc);
    Tensor output(DataType::kDataTypeFp32, size, true, alloc);

    for (int i = 0; i < size; ++i) {
        input.index<float>(i) = 1;
        weight.index<float>(i) = 0.1;
    }

    RMSNormLayer rmsnorm_layer(DeviceType::kDeviceCPU, DataType::kDataTypeFp32);
    LayerParam& layer = rmsnorm_layer;
    layer.reset_weight_size(1);
    layer.set_weight(0, weight);
    layer.forward(input, output);

    ASSERT_EQ(input.size(), size);
    ASSERT_EQ(input.dims_size(), 1);
    ASSERT_EQ(weight.size(), size);
    ASSERT_EQ(input.dims_size(), 1);
    ASSERT_EQ(output.size(), size);
    ASSERT_EQ(input.dims_size(), 1);

    for (int i = 0; i < size; ++i) {
        // LOG(INFO) << " " << output.index<float>(i);
        ASSERT_NEAR(output.index<float>(i), 0.1, 1e-2);
    }
}

TEST(test_rmsnorm, test_gpu_nostream) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto cpu_alloc = CPUDeviceAllocatorFactory::get_instance();
    auto gpu_alloc = CUDADeviceAllocatorFactory::get_instance();

    const int32_t size = 512;
    Tensor input_cpu(DataType::kDataTypeFp32, size, true, cpu_alloc);
    Tensor weight_cpu(DataType::kDataTypeFp32, size, true, cpu_alloc);
    Tensor output_cpu(DataType::kDataTypeFp32, size, true, cpu_alloc);

    std::random_device rd;  // 使用硬件熵源生成真随机数种子
    std::mt19937 mt(rd());  // 用种子初始化 Mersenne Twister 伪随机数引擎
    std::uniform_real_distribution<float> dist(0.f, 1.f);   // 定义均匀分布，范围 [0.0, 1.0)
    for (int i = 0; i < size; ++i) {
        input_cpu.index<float>(i) = dist(mt);
        weight_cpu.index<float>(i) = dist(mt);
    }

    Tensor input_gpu = input_cpu.clone();
    Tensor weight_gpu = weight_cpu.clone();
    Tensor output_gpu = output_cpu.clone();
    input_gpu.to_cuda();
    weight_gpu.to_cuda();
    output_gpu.to_cuda();

    // GPU 计算
    RMSNormLayer rmsnorm_layer_gpu(DeviceType::kDeviceGPU, DataType::kDataTypeFp32);
    LayerParam& layer_gpu = rmsnorm_layer_gpu;
    layer_gpu.set_cuda_config(std::make_shared<kernel::CudaConfig>());
    layer_gpu.reset_weight_size(1);
    layer_gpu.set_weight(0, weight_gpu);
    layer_gpu.forward(input_gpu, output_gpu);

    output_gpu.to_cpu();

    // CPU 计算
    RMSNormLayer rmsnorm_layer_cpu(DeviceType::kDeviceCPU, DataType::kDataTypeFp32);
    LayerParam& layer_cpu = rmsnorm_layer_cpu;
    layer_cpu.reset_weight_size(1);
    layer_cpu.set_weight(0, weight_cpu);
    layer_cpu.forward(input_cpu, output_cpu);

    // 验证
    for (int i = 0; i < size; ++i) {
        // LOG(INFO) << " " << output_cpu.index<float>(i) << " " << output_gpu.index<float>(i);
        ASSERT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 1e-5);
    }
}

TEST(test_rmsnorm, test_gpu_stream) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto cpu_alloc = CPUDeviceAllocatorFactory::get_instance();
    auto gpu_alloc = CUDADeviceAllocatorFactory::get_instance();

    const int32_t size = 512;
    Tensor input_cpu(DataType::kDataTypeFp32, size, true, cpu_alloc);
    Tensor weight_cpu(DataType::kDataTypeFp32, size, true, cpu_alloc);
    Tensor output_cpu(DataType::kDataTypeFp32, size, true, cpu_alloc);

    std::random_device rd;  // 使用硬件熵源生成真随机数种子
    std::mt19937 mt(rd());  // 用种子初始化 Mersenne Twister 伪随机数引擎
    std::uniform_real_distribution<float> dist(0.f, 1.f);   // 定义均匀分布，范围 [0.0, 1.0)
    for (int i = 0; i < size; ++i) {
        input_cpu.index<float>(i) = dist(mt);
        weight_cpu.index<float>(i) = dist(mt);
    }

    Tensor input_gpu = input_cpu.clone();
    Tensor weight_gpu = weight_cpu.clone();
    Tensor output_gpu = output_cpu.clone();
    input_gpu.to_cuda();
    weight_gpu.to_cuda();
    output_gpu.to_cuda();

    // GPU 计算
    RMSNormLayer rmsnorm_layer_gpu(DeviceType::kDeviceGPU, DataType::kDataTypeFp32);
    LayerParam& layer_gpu = rmsnorm_layer_gpu;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    layer_gpu.set_cuda_config(std::make_shared<kernel::CudaConfig>(stream));
    layer_gpu.reset_weight_size(1);
    layer_gpu.set_weight(0, weight_gpu);
    layer_gpu.forward(input_gpu, output_gpu);

    output_gpu.to_cpu();

    // CPU 计算
    RMSNormLayer rmsnorm_layer_cpu(DeviceType::kDeviceCPU, DataType::kDataTypeFp32);
    LayerParam& layer_cpu = rmsnorm_layer_cpu;
    layer_cpu.reset_weight_size(1);
    layer_cpu.set_weight(0, weight_cpu);
    layer_cpu.forward(input_cpu, output_cpu);

    // 验证
    for (int i = 0; i < size; ++i) {
        // LOG(INFO) << " " << output_cpu.index<float>(i) << " " << output_gpu.index<float>(i);
        ASSERT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 1e-5);
    }
}