#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>

#include "op/embedding.h"

TEST(test_embedding, test_cpu) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    const int seq_len = 3;
    const int vocab_size = 8;
    const int dim = 4;

    Tensor input(DataType::kDataTypeInt32, seq_len, true, alloc);
    input.index<int>(0) = 0;
    input.index<int>(1) = 3;
    input.index<int>(2) = 6;

    Tensor weight(DataType::kDataTypeFp32, vocab_size, dim, true, alloc);
    weight.index<float>(0 * dim + 0) = 1;
    weight.index<float>(0 * dim + 1) = 2;
    weight.index<float>(0 * dim + 2) = 3;
    weight.index<float>(0 * dim + 3) = 4;
    weight.index<float>(3 * dim + 0) = 4;
    weight.index<float>(3 * dim + 1) = 3;
    weight.index<float>(3 * dim + 2) = 1;
    weight.index<float>(3 * dim + 3) = 7;
    weight.index<float>(6 * dim + 0) = 6;
    weight.index<float>(6 * dim + 1) = 1;
    weight.index<float>(6 * dim + 2) = 3;
    weight.index<float>(6 * dim + 3) = 4;

    Tensor output(DataType::kDataTypeFp32, seq_len, dim, true, alloc);

    EmbeddingLayer embedding_layer(DeviceType::kDeviceCPU, seq_len, vocab_size, dim);
    Layer& layer = embedding_layer;
    layer.set_weight(0, weight);
    layer.forward(input, output);

    ASSERT_NEAR(output.index<float>(0 * dim + 0), 1, 0.01);
    ASSERT_NEAR(output.index<float>(0 * dim + 1), 2, 0.01);
    ASSERT_NEAR(output.index<float>(0 * dim + 2), 3, 0.01);
    ASSERT_NEAR(output.index<float>(0 * dim + 3), 4, 0.01);
    ASSERT_NEAR(output.index<float>(1 * dim + 0), 4, 0.01);
    ASSERT_NEAR(output.index<float>(1 * dim + 1), 3, 0.01);
    ASSERT_NEAR(output.index<float>(1 * dim + 2), 1, 0.01);
    ASSERT_NEAR(output.index<float>(1 * dim + 3), 7, 0.01);
    ASSERT_NEAR(output.index<float>(2 * dim + 0), 6, 0.01);
    ASSERT_NEAR(output.index<float>(2 * dim + 1), 1, 0.01);
    ASSERT_NEAR(output.index<float>(2 * dim + 2), 3, 0.01);
    ASSERT_NEAR(output.index<float>(2 * dim + 3), 4, 0.01);
}

TEST(test_embedding, test_gpu) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    const int seq_len = 128;
    const int vocab_size = 4096;
    const int dim = 512;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist_int(0, 4095);
    std::uniform_real_distribution<float> dist_float(0.f, 1.f);

    Tensor input_cpu(DataType::kDataTypeInt32, seq_len, true, alloc);
    for (int i = 0; i < input_cpu.size(); ++i) {
        input_cpu.index<int>(i) = dist_int(mt);
    }
    Tensor weight_cpu(DataType::kDataTypeFp32, vocab_size, dim, true, alloc);
    for (int i = 0; i < weight_cpu.size(); ++i) {
        weight_cpu.index<float>(i) = dist_float(mt);
    }
    Tensor output_cpu(DataType::kDataTypeFp32, dim, true, alloc);

    // GPU 计算
    Tensor input_gpu = input_cpu.clone();
    Tensor weight_gpu = weight_cpu.clone();
    Tensor output_gpu = output_cpu.clone();
    input_gpu.to_cuda();
    weight_gpu.to_cuda();
    output_gpu.to_cuda();

    EmbeddingLayer embedding_layer_gpu(DeviceType::kDeviceGPU, seq_len, vocab_size, dim);
    Layer& layer_gpu = embedding_layer_gpu;
    layer_gpu.set_cuda_config(std::make_shared<kernel::CudaConfig>());
    layer_gpu.set_weight(0, weight_gpu);
    layer_gpu.forward(input_gpu, output_gpu);

    output_gpu.to_cpu();

    // CPU 计算
    EmbeddingLayer embedding_layer_cpu(DeviceType::kDeviceCPU, seq_len, vocab_size, dim);
    Layer& layer_cpu = embedding_layer_cpu;
    layer_cpu.set_weight(0, weight_cpu);
    layer_cpu.forward(input_cpu, output_cpu);

    // 验证
    for (int i = 0; i < output_cpu.size(); ++i) {
        ASSERT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 1e-2);
    }
}