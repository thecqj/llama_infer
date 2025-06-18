#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>

#include "../source/op/kernel/cpu/softmax_kernel.h"
#include "../source/op/kernel/cpu/scale_sum_kernel.h"
#include "op/mha.h"

TEST(test_softmax, test_cpu) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    Tensor input(DataType::kDataTypeFp32, 4, true, alloc);
    input.index<float>(0) = 3.6f;
    input.index<float>(1) = 2.5f;
    input.index<float>(2) = 1.1f;
    input.index<float>(3) = 4.4f;

    kernel::softmax_inplace_cpu(input);

    ASSERT_NEAR(input.index<float>(0), 0.2747, 0.01);
    ASSERT_NEAR(input.index<float>(1), 0.0914, 0.01);
    ASSERT_NEAR(input.index<float>(2), 0.0225, 0.01);
    ASSERT_NEAR(input.index<float>(3), 0.6113, 0.01);
}

TEST(test_scale_sum, test_cpu) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    Tensor scale(DataType::kDataTypeFp32, 4, true, alloc);
    scale.index<float>(0) = 0.2747;
    scale.index<float>(1) = 0.0914;
    scale.index<float>(2) = 0.0225;
    scale.index<float>(3) = 0.6113;

    Tensor value(DataType::kDataTypeFp32, 4, 5, true, alloc);
    for (int i = 0; i < value.size(); ++i) {
        value.index<float>(i) = i + 1;
    }

    Tensor output(DataType::kDataTypeFp32, 5, true, alloc);
    for (int i = 0; i < output.size(); ++i) {
        output.index<float>(i) = 0;
    }

    kernel::scale_sum_kernel_cpu(value, scale, output, 3, 5, 5);

    ASSERT_NEAR(output.index<float>(0), 10.8514, 0.01);
    ASSERT_NEAR(output.index<float>(1), 11.8513, 0.01);
    ASSERT_NEAR(output.index<float>(2), 12.8512, 0.01);
    ASSERT_NEAR(output.index<float>(3), 13.8511, 0.01);
    ASSERT_NEAR(output.index<float>(4), 14.8510, 0.01);
}

TEST(test_mha, test_cpu) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    const int layer_index = 0;
    const int pos = 1;
    const int kv_mul = 2;
    const int kv_dim = 8;
    const int seq_len = 4;
    const int head_size = 4;
    const int head_num = 4;

    Tensor query(DataType::kDataTypeFp32, head_num, head_size, true, alloc);
    for (int i = 0; i < query.size(); ++i) {
        query.index<float>(i) = (i % 4) + 1;
    }

    Tensor key(DataType::kDataTypeFp32, pos + 1, head_num / kv_mul, head_size, true, alloc);
    key.index<float>(0) = 3; key.index<float>(1) = 4; key.index<float>(2) = 2; key.index<float>(3) = 8;
    key.index<float>(4) = 2; key.index<float>(5) = 5; key.index<float>(6) = 2; key.index<float>(7) = 7;
    key.index<float>(8) = 2; key.index<float>(9) = 5; key.index<float>(10) = 2; key.index<float>(11) = 7;
    key.index<float>(12) = 3; key.index<float>(13) = 4; key.index<float>(14) = 2; key.index<float>(15) = 8;

    Tensor score(DataType::kDataTypeFp32, head_num, seq_len, true, alloc);
    for (int i = 0; i < score.size(); ++i) {
        score.index<float>(i) = 0;
    }

    Tensor value(DataType::kDataTypeFp32, pos + 1, head_num / kv_mul, head_size, true, alloc);
    value.index<float>(0) = 1; value.index<float>(1) = 2; value.index<float>(2) = 3; value.index<float>(3) = 4;
    value.index<float>(4) = 1; value.index<float>(5) = 2; value.index<float>(6) = 3; value.index<float>(7) = 4;
    value.index<float>(8) = 4; value.index<float>(9) = 3; value.index<float>(10) = 2; value.index<float>(11) = 1;
    value.index<float>(12) = 4; value.index<float>(13) = 3; value.index<float>(14) = 2; value.index<float>(15) = 1;

    Tensor output(DataType::kDataTypeFp32, head_num, head_size, true, alloc);
    for (int i = 0; i < output.size(); ++i) {
        output.index<float>(i) = 0;
    }

    MultiHeadAttentionLayer mha_layer(DeviceType::kDeviceCPU, layer_index, kv_mul, kv_dim, seq_len, head_num, head_size);
    mha_layer.set_pos(pos);
    Layer& layer = mha_layer;
    layer.set_input(0, query);
    layer.set_input(1, score);
    layer.set_input(2, key);
    layer.set_input(3, value);
    layer.set_output(0, output);
    layer.forward();

    ASSERT_NEAR(score.index<float>(0), 0.8176, 0.01);
    ASSERT_NEAR(score.index<float>(1), 0.1824, 0.01);
    ASSERT_NEAR(score.index<float>(2), 0, 0.01);
    ASSERT_NEAR(score.index<float>(3), 0, 0.01);
    ASSERT_NEAR(score.index<float>(4), 0.8176, 0.01);
    ASSERT_NEAR(score.index<float>(5), 0.1824, 0.01);
    ASSERT_NEAR(score.index<float>(6), 0, 0.01);
    ASSERT_NEAR(score.index<float>(7), 0, 0.01);
    ASSERT_NEAR(score.index<float>(8), 0.1824, 0.01);
    ASSERT_NEAR(score.index<float>(9), 0.8176, 0.01);
    ASSERT_NEAR(score.index<float>(10), 0, 0.01);
    ASSERT_NEAR(score.index<float>(11), 0, 0.01);
    ASSERT_NEAR(score.index<float>(12), 0.1824, 0.01);
    ASSERT_NEAR(score.index<float>(13), 0.8176, 0.01);
    ASSERT_NEAR(score.index<float>(14), 0, 0.01);
    ASSERT_NEAR(score.index<float>(15), 0, 0.01);

    ASSERT_NEAR(output.index<float>(0), 1.5473, 0.01);
    ASSERT_NEAR(output.index<float>(1), 2.1824, 0.01);
    ASSERT_NEAR(output.index<float>(2), 2.8176, 0.01);
    ASSERT_NEAR(output.index<float>(3), 3.4527, 0.01);
    ASSERT_NEAR(output.index<float>(4), 1.5473, 0.01);
    ASSERT_NEAR(output.index<float>(5), 2.1824, 0.01);
    ASSERT_NEAR(output.index<float>(6), 2.8176, 0.01);
    ASSERT_NEAR(output.index<float>(7), 3.4527, 0.01);
    ASSERT_NEAR(output.index<float>(8), 3.4527, 0.01);
    ASSERT_NEAR(output.index<float>(9), 2.8176, 0.01);
    ASSERT_NEAR(output.index<float>(10), 2.1824, 0.01);
    ASSERT_NEAR(output.index<float>(11), 1.5473, 0.01);
    ASSERT_NEAR(output.index<float>(12), 3.4527, 0.01);
    ASSERT_NEAR(output.index<float>(13), 2.8176, 0.01);
    ASSERT_NEAR(output.index<float>(14), 2.1824, 0.01);
    ASSERT_NEAR(output.index<float>(15), 1.5473, 0.01);
}

TEST(test_mha, test_gpu) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto cpu_alloc = CPUDeviceAllocatorFactory::get_instance();
    auto gpu_alloc = CUDADeviceAllocatorFactory::get_instance();

    const int layer_index = 0;
    const int pos = 1;
    const int kv_mul = 2;
    const int kv_dim = 8;
    const int seq_len = 4;
    const int head_size = 4;
    const int head_num = 4;

    Tensor query_cpu(DataType::kDataTypeFp32, head_num, head_size, true, cpu_alloc);
    for (int i = 0; i < query_cpu.size(); ++i) {
        query_cpu.index<float>(i) = (i % 4) + 1;
    }

    Tensor key_cpu(DataType::kDataTypeFp32, pos + 1, head_num / kv_mul, head_size, true, cpu_alloc);
    key_cpu.index<float>(0) = 3; key_cpu.index<float>(1) = 4; key_cpu.index<float>(2) = 2; key_cpu.index<float>(3) = 8;
    key_cpu.index<float>(4) = 2; key_cpu.index<float>(5) = 5; key_cpu.index<float>(6) = 2; key_cpu.index<float>(7) = 7;
    key_cpu.index<float>(8) = 2; key_cpu.index<float>(9) = 5; key_cpu.index<float>(10) = 2; key_cpu.index<float>(11) = 7;
    key_cpu.index<float>(12) = 3; key_cpu.index<float>(13) = 4; key_cpu.index<float>(14) = 2; key_cpu.index<float>(15) = 8;

    Tensor score_cpu(DataType::kDataTypeFp32, head_num, seq_len, true, cpu_alloc);
    for (int i = 0; i < score_cpu.size(); ++i) {
        score_cpu.index<float>(i) = 0;
    }

    Tensor value_cpu(DataType::kDataTypeFp32, pos + 1, head_num / kv_mul, head_size, true, cpu_alloc);
    value_cpu.index<float>(0) = 1; value_cpu.index<float>(1) = 2; value_cpu.index<float>(2) = 3; value_cpu.index<float>(3) = 4;
    value_cpu.index<float>(4) = 1; value_cpu.index<float>(5) = 2; value_cpu.index<float>(6) = 3; value_cpu.index<float>(7) = 4;
    value_cpu.index<float>(8) = 4; value_cpu.index<float>(9) = 3; value_cpu.index<float>(10) = 2; value_cpu.index<float>(11) = 1;
    value_cpu.index<float>(12) = 4; value_cpu.index<float>(13) = 3; value_cpu.index<float>(14) = 2; value_cpu.index<float>(15) = 1;

    Tensor output_cpu(DataType::kDataTypeFp32, head_num, head_size, true, cpu_alloc);
    for (int i = 0; i < output_cpu.size(); ++i) {
        output_cpu.index<float>(i) = 0;
    }

    Tensor query_gpu = query_cpu.clone();
    Tensor key_gpu = key_cpu.clone();
    Tensor score_gpu = score_cpu.clone();
    Tensor value_gpu = value_cpu.clone();
    Tensor output_gpu = output_cpu.clone();
    query_gpu.to_cuda();
    key_gpu.to_cuda();
    score_gpu.to_cuda();
    value_gpu.to_cuda();
    output_gpu.to_cuda();

    // GPU计算
    MultiHeadAttentionLayer mha_layer_gpu(DeviceType::kDeviceGPU, layer_index, kv_mul, kv_dim, seq_len, head_num, head_size);
    mha_layer_gpu.set_pos(pos);
    Layer& layer_gpu = mha_layer_gpu;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    layer_gpu.set_cuda_config(std::make_shared<kernel::CudaConfig>(stream));
    layer_gpu.set_input(0, query_gpu);
    layer_gpu.set_input(1, score_gpu);
    layer_gpu.set_input(2, key_gpu);
    layer_gpu.set_input(3, value_gpu);
    layer_gpu.set_output(0, output_gpu);
    layer_gpu.forward();

    score_gpu.to_cpu();
    output_gpu.to_cpu();

    // CPU计算
    MultiHeadAttentionLayer mha_layer_cpu(DeviceType::kDeviceCPU, layer_index, kv_mul, kv_dim, seq_len, head_num, head_size);
    mha_layer_cpu.set_pos(pos);
    Layer& layer_cpu = mha_layer_cpu;
    layer_cpu.set_input(0, query_cpu);
    layer_cpu.set_input(1, score_cpu);
    layer_cpu.set_input(2, key_cpu);
    layer_cpu.set_input(3, value_cpu);
    layer_cpu.set_output(0, output_cpu);
    layer_cpu.forward();

    // 验证
    for (int i = 0; i < score_cpu.size(); ++i) {
        // LOG(INFO) << "cpu-> " << score_cpu.index<float>(i) << ", gpu-> " << score_gpu.index<float>(i);
        ASSERT_NEAR(score_cpu.index<float>(i), score_gpu.index<float>(i), 0.01);
    }

    for (int i = 0; i < output_cpu.size(); ++i) {
        // LOG(INFO) << "cpu-> " << output_cpu.index<float>(i) << ", gpu-> " << output_gpu.index<float>(i);
        ASSERT_NEAR(output_cpu.index<float>(i), output_gpu.index<float>(i), 0.01);
    }
}