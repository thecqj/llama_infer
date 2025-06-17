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

    for (int i = 0; i < score.size(); ++i) {
        LOG(INFO) << score.index<float>(i);
    }

    for (int i = 0; i < output.size(); ++i) {
        LOG(INFO) << output.index<float>(i);
    }

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