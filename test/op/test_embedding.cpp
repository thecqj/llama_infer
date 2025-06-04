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