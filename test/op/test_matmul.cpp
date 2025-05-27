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