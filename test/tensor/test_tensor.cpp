#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/alloc.h"
#include "tensor/tensor.h"

// 空tensor
TEST(test_tensor, constructor1_alloc_0) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    Tensor tensor(DataType::kDataTypeInt32, 0);
    CHECK_EQ(tensor.empty(), true);
    CHECK_EQ(tensor.size(), 0);
    CHECK_EQ(tensor.byte_size(), 0);
    CHECK_EQ(tensor.dims_size(), 1);
    CHECK_EQ(tensor.get_dim(0), 0);
    CHECK_EQ(tensor.get_buffer(), nullptr);
}

// 使用分配器创建tensor
TEST(test_tensor, constructor1_alloc_1) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    Tensor tensor(DataType::kDataTypeInt32, 2, true, alloc);
    CHECK_EQ(tensor.empty(), false);
    CHECK_EQ(tensor.size(), 2);
    CHECK_EQ(tensor.byte_size(), 8);
    CHECK_EQ(tensor.dims_size(), 1);
    CHECK_EQ(tensor.get_dim(0), 2);
    CHECK_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    CHECK_EQ(buffer->byte_size(), 8);
    CHECK_EQ(buffer->is_externel(), false);
}

// 不使用分配器创建tensor
TEST(test_tensor, constructor1_externel) {
    using namespace tensor;
    using namespace base;

    float* ptr = new float{5};
    Tensor tensor(DataType::kDataTypeFp32, 1, false, nullptr, ptr);
    CHECK_EQ(tensor.empty(), false);
    CHECK_EQ(tensor.size(), 1);
    CHECK_EQ(tensor.byte_size(), 4);
    CHECK_EQ(tensor.dims_size(), 1);
    CHECK_EQ(tensor.get_dim(0), 1);
    CHECK_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    CHECK_EQ(buffer->byte_size(), 4);
    CHECK_EQ(buffer->is_externel(), true);
    CHECK_EQ(*static_cast<float*>(buffer->ptr()), *ptr);
}

// 使用分配器创建tensor
TEST(test_tensor, constructor2_alloc) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    Tensor tensor(DataType::kDataTypeInt32, 2, 5, true, alloc);
    CHECK_EQ(tensor.empty(), false);
    CHECK_EQ(tensor.size(), 10);
    CHECK_EQ(tensor.byte_size(), 40);
    CHECK_EQ(tensor.dims_size(), 2);
    CHECK_EQ(tensor.get_dim(0), 2);
    CHECK_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    CHECK_EQ(buffer->byte_size(), 40);
    CHECK_EQ(buffer->is_externel(), false);
}

// 不使用分配器创建tensor
TEST(test_tensor, constructor2_externel) {
    using namespace tensor;
    using namespace base;

    float* ptr = new float[10];
    *ptr = 3.14;
    Tensor tensor(DataType::kDataTypeFp32, 2, 5, false, nullptr, ptr);
    CHECK_EQ(tensor.empty(), false);
    CHECK_EQ(tensor.size(), 10);
    CHECK_EQ(tensor.byte_size(), 40);
    CHECK_EQ(tensor.dims_size(), 2);
    CHECK_EQ(tensor.get_dim(0), 2);
    CHECK_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    CHECK_EQ(buffer->byte_size(), 40);
    CHECK_EQ(buffer->is_externel(), true);
    CHECK_EQ(*static_cast<float*>(buffer->ptr()), *ptr);
}

// 使用分配器创建tensor
TEST(test_tensor, constructor3_alloc) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    Tensor tensor(DataType::kDataTypeInt32, 2, 2, 5, true, alloc);
    CHECK_EQ(tensor.empty(), false);
    CHECK_EQ(tensor.size(), 20);
    CHECK_EQ(tensor.byte_size(), 80);
    CHECK_EQ(tensor.dims_size(), 3);
    CHECK_EQ(tensor.get_dim(2), 5);
    CHECK_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    CHECK_EQ(buffer->byte_size(), 80);
    CHECK_EQ(buffer->is_externel(), false);
}

// 不使用分配器创建tensor
TEST(test_tensor, constructor3_externel) {
    using namespace tensor;
    using namespace base;

    float* ptr = new float{20};
    *ptr = 3.14;
    Tensor tensor(DataType::kDataTypeFp32, 2, 2, 5, false, nullptr, ptr);
    CHECK_EQ(tensor.empty(), false);
    CHECK_EQ(tensor.size(), 20);
    CHECK_EQ(tensor.byte_size(), 80);
    CHECK_EQ(tensor.dims_size(), 3);
    CHECK_EQ(tensor.get_dim(1), 2);
    CHECK_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    CHECK_EQ(buffer->byte_size(), 80);
    CHECK_EQ(buffer->is_externel(), true);
    CHECK_EQ(*static_cast<float*>(buffer->ptr()), *ptr);
}

// 使用分配器创建tensor
TEST(test_tensor, constructor4_alloc) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    Tensor tensor(DataType::kDataTypeInt32, 1, 2, 2, 5, true, alloc);
    CHECK_EQ(tensor.empty(), false);
    CHECK_EQ(tensor.size(), 20);
    CHECK_EQ(tensor.byte_size(), 80);
    CHECK_EQ(tensor.dims_size(), 4);
    CHECK_EQ(tensor.get_dim(3), 5);
    CHECK_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    CHECK_EQ(buffer->byte_size(), 80);
    CHECK_EQ(buffer->is_externel(), false);
}

// 不使用分配器创建tensor
TEST(test_tensor, constructor4_externel) {
    using namespace tensor;
    using namespace base;

    float* ptr = new float{20};
    *ptr = 3.14;
    Tensor tensor(DataType::kDataTypeFp32, 1, 2, 2, 5, false, nullptr, ptr);
    CHECK_EQ(tensor.empty(), false);
    CHECK_EQ(tensor.size(), 20);
    CHECK_EQ(tensor.byte_size(), 80);
    CHECK_EQ(tensor.dims_size(), 4);
    CHECK_EQ(tensor.get_dim(0), 1);
    CHECK_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    CHECK_EQ(buffer->byte_size(), 80);
    CHECK_EQ(buffer->is_externel(), true);
    CHECK_EQ(*static_cast<float*>(buffer->ptr()), *ptr);
}

// 使用分配器创建tensor
TEST(test_tensor, constructor5_alloc) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{1, 2, 2, 5};

    Tensor tensor(DataType::kDataTypeInt32, dims, true, alloc);
    CHECK_EQ(tensor.empty(), false);
    CHECK_EQ(tensor.size(), 20);
    CHECK_EQ(tensor.byte_size(), 80);
    CHECK_EQ(tensor.dims_size(), 4);
    CHECK_EQ(tensor.get_dim(3), 5);
    CHECK_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    CHECK_EQ(buffer->byte_size(), 80);
    CHECK_EQ(buffer->is_externel(), false);
}

// 不使用分配器创建tensor
TEST(test_tensor, constructor5_externel) {
    using namespace tensor;
    using namespace base;
    std::vector<int32_t> dims{1, 2, 2, 5};

    float* ptr = new float{20};
    *ptr = 3.14;
    Tensor tensor(DataType::kDataTypeFp32, dims, false, nullptr, ptr);
    CHECK_EQ(tensor.empty(), false);
    CHECK_EQ(tensor.size(), 20);
    CHECK_EQ(tensor.byte_size(), 80);
    CHECK_EQ(tensor.dims_size(), 4);
    CHECK_EQ(tensor.get_dim(0), 1);
    CHECK_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    CHECK_EQ(buffer->byte_size(), 80);
    CHECK_EQ(buffer->is_externel(), true);
    CHECK_EQ(*static_cast<float*>(buffer->ptr()), *ptr);
}