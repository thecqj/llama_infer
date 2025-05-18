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

    delete ptr;
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

    delete[] ptr;
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

    delete[] ptr;
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

    delete[] ptr;
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

    delete[] ptr;
}

TEST(test_tensor, ptr_alloc) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{1, 2, 2, 5};

    Tensor tensor(DataType::kDataTypeInt32, dims, true, alloc);
    auto ptr = tensor.ptr<int32_t>();
    alloc->memset_zero(ptr, tensor.byte_size());

    *tensor.ptr<int32_t>() = 3;
    *tensor.ptr<int32_t>(1) = 5;
    tensor.index<int32_t>(2) = 10;

    CHECK_EQ(*ptr, 3);
    CHECK_EQ(*(ptr + 1), 5);
    CHECK_EQ(*(ptr + 2), 10);
}

TEST(test_tensor, ptr_externel) {
    using namespace tensor;
    using namespace base;
    std::vector<int32_t> dims{1, 2, 2, 5};

    float* ptr = new float{20};
    *ptr = 3.14;
    *(ptr + 1) = 5;
    *(ptr + 2) = 10.55;
    const Tensor tensor(DataType::kDataTypeFp32, dims, false, nullptr, ptr);

    ASSERT_FLOAT_EQ(*tensor.ptr<float>(), 3.14);
    ASSERT_FLOAT_EQ(*tensor.ptr<float>(1), 5);
    ASSERT_FLOAT_EQ(tensor.index<float>(2), 10.55);
}

TEST(test_tensor, strides) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{1, 2, 2, 5};

    Tensor tensor(DataType::kDataTypeInt32, dims, true, alloc);
    auto strides = tensor.strides();

    ASSERT_EQ(strides[0], 20);
    ASSERT_EQ(strides[1], 10);
    ASSERT_EQ(strides[2], 5);
    ASSERT_EQ(strides[3], 1);
}

TEST(test_tensor, reshape_0) {
    using namespace tensor;
    using namespace base;

    Tensor tensor(DataType::kDataTypeInt32, 0); // 空张量

    std::vector<int32_t> new_dims{1, 2, 2, 5};
    tensor.reshape(new_dims);

    ASSERT_EQ(tensor.empty(), true);
    ASSERT_EQ(tensor.size(), 20);
    ASSERT_EQ(tensor.byte_size(), 80);
    ASSERT_EQ(tensor.dims_size(), 4);
    ASSERT_EQ(tensor.get_dim(2), 2);
    ASSERT_EQ(tensor.get_buffer(), nullptr);
}

TEST(test_tensor, reshape_1) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{2, 5};

    Tensor tensor(DataType::kDataTypeInt32, dims, true, alloc);
    auto buffer_old = tensor.get_buffer();

    std::vector<int32_t> new_dims{1, 2, 2, 5};  // 扩容
    tensor.reshape(new_dims);

    ASSERT_EQ(tensor.empty(), false);
    ASSERT_EQ(tensor.size(), 20);
    ASSERT_EQ(tensor.byte_size(), 80);
    ASSERT_EQ(tensor.dims_size(), 4);
    ASSERT_EQ(tensor.get_dim(2), 2);
    ASSERT_NE(tensor.get_buffer(), buffer_old);

    auto buffer_new = tensor.get_buffer();
    ASSERT_EQ(buffer_new->byte_size(), 80);
    ASSERT_EQ(buffer_new->is_externel(), false);
}

TEST(test_tensor, reshape_2) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{1, 2, 2, 5};

    Tensor tensor(DataType::kDataTypeInt32, dims, true, alloc);
    auto buffer_old = tensor.get_buffer();

    std::vector<int32_t> new_dims{2, 5};    // 不需要扩容
    tensor.reshape(new_dims);

    ASSERT_EQ(tensor.empty(), false);
    ASSERT_EQ(tensor.size(), 10);
    ASSERT_EQ(tensor.byte_size(), 40);
    ASSERT_EQ(tensor.dims_size(), 2);
    ASSERT_EQ(tensor.get_dim(1), 5);
    ASSERT_EQ(tensor.get_buffer(), buffer_old);

    auto buffer_new = tensor.get_buffer();
    ASSERT_EQ(buffer_new->byte_size(), 80);
    ASSERT_EQ(buffer_new->is_externel(), false);
}

TEST(test_tensor, clone_0) {
    using namespace tensor;
    using namespace base;

    Tensor tensor(DataType::kDataTypeInt32, 0); // 空张量
    Tensor new_tensor = tensor.clone();

    ASSERT_EQ(new_tensor.empty(), true);
    ASSERT_EQ(new_tensor.size(), 0);
    ASSERT_EQ(new_tensor.byte_size(), 0);
    ASSERT_EQ(new_tensor.dims_size(), 1);
    ASSERT_EQ(new_tensor.get_dim(0), 0);
    ASSERT_EQ(new_tensor.get_buffer(), nullptr);
}

TEST(test_tensor, clone_1) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{1, 2, 2, 5};

    Tensor tensor(DataType::kDataTypeInt32, dims, true, alloc); // 具有内存所属权的张量
    Tensor new_tensor = tensor.clone();

    ASSERT_EQ(new_tensor.empty(), false);
    ASSERT_EQ(new_tensor.size(), 20);
    ASSERT_EQ(new_tensor.byte_size(), 80);
    ASSERT_EQ(new_tensor.dims_size(), 4);
    ASSERT_EQ(new_tensor.get_dim(0), 1);
    ASSERT_NE(new_tensor.get_buffer(), tensor.get_buffer());

    auto buffer = new_tensor.get_buffer();
    ASSERT_EQ(buffer->byte_size(), 80);
    ASSERT_EQ(buffer->is_externel(), false);
}

TEST(test_tensor, clone_2) {
    using namespace tensor;
    using namespace base;
    std::vector<int32_t> dims{1, 2, 2, 5};

    float* ptr = new float[20];
    Tensor tensor(DataType::kDataTypeInt32, dims, false, nullptr, ptr, base::DeviceType::kDeviceCPU); // 不具有内存所属权的张量
    Tensor new_tensor = tensor.clone();

    ASSERT_EQ(new_tensor.empty(), false);
    ASSERT_EQ(new_tensor.size(), 20);
    ASSERT_EQ(new_tensor.byte_size(), 80);
    ASSERT_EQ(new_tensor.dims_size(), 4);
    ASSERT_EQ(new_tensor.get_dim(0), 1);
    ASSERT_NE(new_tensor.get_buffer(), tensor.get_buffer());

    auto buffer = new_tensor.get_buffer();
    ASSERT_EQ(buffer->byte_size(), 80);
    ASSERT_EQ(buffer->is_externel(), false);

    delete[] ptr;
}

TEST(test_tensor, to_cpu) {
    using namespace tensor;
    using namespace base;
    auto alloc = CUDADeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{1, 2, 2, 5};

    Tensor tensor(DataType::kDataTypeFp32, dims, true, alloc);
    ASSERT_EQ(tensor.device_type(), DeviceType::kDeviceGPU);

    tensor.to_cpu();
    ASSERT_EQ(tensor.device_type(), DeviceType::kDeviceCPU);
    ASSERT_EQ(tensor.empty(), false);
    ASSERT_EQ(tensor.size(), 20);
    ASSERT_EQ(tensor.byte_size(), 80);
    ASSERT_EQ(tensor.dims_size(), 4);
    ASSERT_EQ(tensor.get_dim(0), 1);
    ASSERT_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    ASSERT_EQ(buffer->byte_size(), 80);
    ASSERT_EQ(buffer->is_externel(), false);
}

TEST(test_tensor, to_cpu_externel) {
    using namespace tensor;
    using namespace base;

    float* d_ptr;
    const size_t byte_size = 20 * sizeof(float);
    cudaMalloc(&d_ptr, byte_size);

    std::vector<int32_t> dims{1, 2, 2, 5};
    Tensor tensor(DataType::kDataTypeFp32, dims, false, nullptr, d_ptr, DeviceType::kDeviceGPU);
    ASSERT_EQ(tensor.device_type(), DeviceType::kDeviceGPU);

    tensor.to_cpu();
    ASSERT_EQ(tensor.device_type(), DeviceType::kDeviceCPU);
    ASSERT_EQ(tensor.empty(), false);
    ASSERT_EQ(tensor.size(), 20);
    ASSERT_EQ(tensor.byte_size(), 80);
    ASSERT_EQ(tensor.dims_size(), 4);
    ASSERT_EQ(tensor.get_dim(0), 1);
    ASSERT_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    ASSERT_EQ(buffer->byte_size(), 80);
    ASSERT_EQ(buffer->is_externel(), false);

    cudaFree(d_ptr);
}

TEST(test_tensor, to_cuda) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{1, 2, 2, 5};

    Tensor tensor(DataType::kDataTypeFp32, dims, true, alloc);
    ASSERT_EQ(tensor.device_type(), DeviceType::kDeviceCPU);

    tensor.to_cuda();
    ASSERT_EQ(tensor.device_type(), DeviceType::kDeviceGPU);
    ASSERT_EQ(tensor.empty(), false);
    ASSERT_EQ(tensor.size(), 20);
    ASSERT_EQ(tensor.byte_size(), 80);
    ASSERT_EQ(tensor.dims_size(), 4);
    ASSERT_EQ(tensor.get_dim(0), 1);
    ASSERT_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    ASSERT_EQ(buffer->byte_size(), 80);
    ASSERT_EQ(buffer->is_externel(), false);
}

TEST(test_tensor, to_cuda_externel) {
    using namespace tensor;
    using namespace base;

    const size_t byte_size = 20 * sizeof(float);
    float* h_ptr = (float*)malloc(byte_size);

    std::vector<int32_t> dims{1, 2, 2, 5};
    Tensor tensor(DataType::kDataTypeFp32, dims, false, nullptr, h_ptr, DeviceType::kDeviceCPU);
    ASSERT_EQ(tensor.device_type(), DeviceType::kDeviceCPU);

    tensor.to_cuda();
    ASSERT_EQ(tensor.device_type(), DeviceType::kDeviceGPU);
    ASSERT_EQ(tensor.empty(), false);
    ASSERT_EQ(tensor.size(), 20);
    ASSERT_EQ(tensor.byte_size(), 80);
    ASSERT_EQ(tensor.dims_size(), 4);
    ASSERT_EQ(tensor.get_dim(0), 1);
    ASSERT_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    ASSERT_EQ(buffer->byte_size(), 80);
    ASSERT_EQ(buffer->is_externel(), false);

    free(h_ptr);
}

TEST(test_tensor, assign) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{1, 2, 2, 5};

    Tensor tensor(DataType::kDataTypeInt32, dims, true, alloc);
    auto buffer_old = tensor.get_buffer();

    auto buffer_new = std::make_shared<Buffer>();
    ASSERT_EQ(tensor.assign(buffer_new), false);
    ASSERT_EQ(tensor.get_buffer(), buffer_old);

    buffer_new = std::make_shared<Buffer>(40, alloc);
    ASSERT_EQ(tensor.assign(buffer_new), false);
    ASSERT_EQ(tensor.get_buffer(), buffer_old);

    buffer_new = std::make_shared<Buffer>(160, alloc);
    ASSERT_EQ(tensor.assign(buffer_new), true);
    ASSERT_NE(tensor.get_buffer(), buffer_old);
}

TEST(test_tensor, reset) {
    using namespace tensor;
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{1, 2, 2, 5};

    Tensor tensor(DataType::kDataTypeInt32, dims, true, alloc);
    tensor.reset(DataType::kDataTypeFp32, {2, 2});

    ASSERT_EQ(tensor.empty(), true);
    ASSERT_EQ(tensor.size(), 4);
    ASSERT_EQ(tensor.byte_size(), 16);
    ASSERT_EQ(tensor.dims_size(), 2);
    ASSERT_EQ(tensor.get_dim(0), 2);
    ASSERT_EQ(tensor.get_buffer(), nullptr);

    tensor.allocate(alloc, true);
    ASSERT_NE(tensor.get_buffer(), nullptr);

    auto buffer = tensor.get_buffer();
    ASSERT_EQ(buffer->byte_size(), 16);
    ASSERT_EQ(buffer->is_externel(), false);
}