#include <glog/logging.h>
#include <gtest/gtest.h>

#include "op/add.h"

TEST(test_add, add_cpu_1d) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{64};

    Tensor input1(DataType::kDataTypeFp32, dims, true, alloc);
    auto ptr1 = input1.ptr<float>();
    for (int i = 0; i < input1.size(); ++i) {
        *ptr1++ = i;
    }

    Tensor input2(DataType::kDataTypeFp32, dims, true, alloc);
    auto ptr2 = input2.ptr<float>();
    for (int i = 0; i < input2.size(); ++i) {
        *ptr2++ = i;
    }

    Tensor output(DataType::kDataTypeFp32, dims, true, alloc);

    VecAddLayer add_layer(DeviceType::kDeviceCPU);
    Layer& layer = add_layer;
    layer.forward(input1, input2, output);

    ASSERT_EQ(input1.size(), 64);
    ASSERT_EQ(input1.dims_size(), 1);
    ASSERT_EQ(input2.size(), 64);
    ASSERT_EQ(input2.dims_size(), 1);
    ASSERT_EQ(output.size(), 64);
    ASSERT_EQ(output.dims_size(), 1);

    auto ptr = output.ptr<float>();
    for (int i = 0; i < output.size(); ++i) {
        ASSERT_FLOAT_EQ(*ptr++, 2.0f * i);
    }
}

TEST(test_add, add_cpu_2d) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{2, 32};

    Tensor input1(DataType::kDataTypeFp32, dims, true, alloc);
    auto ptr1 = input1.ptr<float>();
    for (int i = 0; i < input1.size(); ++i) {
        *ptr1++ = i;
    }

    Tensor input2(DataType::kDataTypeFp32, dims, true, alloc);
    auto ptr2 = input2.ptr<float>();
    for (int i = 0; i < input2.size(); ++i) {
        *ptr2++ = i;
    }

    Tensor output(DataType::kDataTypeFp32, dims, true, alloc);

    VecAddLayer add_layer(DeviceType::kDeviceCPU);
    Layer& layer = add_layer;
    layer.forward(input1, input2, output);

    ASSERT_EQ(input1.size(), 64);
    ASSERT_EQ(input1.dims_size(), 2);
    ASSERT_EQ(input2.size(), 64);
    ASSERT_EQ(input2.dims_size(), 2);
    ASSERT_EQ(output.size(), 64);
    ASSERT_EQ(output.dims_size(), 2);

    auto ptr = output.ptr<float>();
    for (int i = 0; i < output.size(); ++i) {
        ASSERT_FLOAT_EQ(*ptr++, 2.0f * i);
    }
}

TEST(test_add, add_gpu_nostream) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CUDADeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{64};

    float* ptr1 = new float[64];
    float* ptr2 = new float[64];
    for (int i = 0; i < 64; ++i) {
        ptr1[i] = 1.f;
        ptr2[i] = 2.f;
    }

    Tensor input1(DataType::kDataTypeFp32, dims, true, alloc);
    alloc->memcpy(input1.ptr<float>(), ptr1, input1.byte_size(), MemcpyKind::kMemcpyCPU2CUDA);

    Tensor input2(DataType::kDataTypeFp32, dims, true, alloc);
    alloc->memcpy(input2.ptr<float>(), ptr2, input2.byte_size(), MemcpyKind::kMemcpyCPU2CUDA);

    Tensor output(DataType::kDataTypeFp32, dims, true, alloc);

    VecAddLayer add_layer(DeviceType::kDeviceGPU);
    Layer& layer = add_layer;
    layer.set_cuda_config(std::make_shared<kernel::CudaConfig>());

    ASSERT_EQ(layer.device_type(), DeviceType::kDeviceGPU);
    layer.forward(input1, input2, output);

    ASSERT_EQ(input1.size(), 64);
    ASSERT_EQ(input1.dims_size(), 1);
    ASSERT_EQ(input2.size(), 64);
    ASSERT_EQ(input2.dims_size(), 1);
    ASSERT_EQ(output.size(), 64);
    ASSERT_EQ(output.dims_size(), 1);

    output.to_cpu();
    ASSERT_EQ(output.device_type(), DeviceType::kDeviceCPU);

    float* ptr = output.ptr<float>();
    for (int i = 0; i < output.size(); ++i) {
        ASSERT_FLOAT_EQ(ptr[i], 3.f);
    }

    delete[] ptr1;
    delete[] ptr2;
}

TEST(test_add, add_gpu_stream) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    auto alloc = CUDADeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{64};

    float* ptr1 = new float[64];
    float* ptr2 = new float[64];
    for (int i = 0; i < 64; ++i) {
        ptr1[i] = 1.f;
        ptr2[i] = 2.f;
    }

    Tensor input1(DataType::kDataTypeFp32, dims, true, alloc);
    alloc->memcpy(input1.ptr<float>(), ptr1, input1.byte_size(), MemcpyKind::kMemcpyCPU2CUDA);

    Tensor input2(DataType::kDataTypeFp32, dims, true, alloc);
    alloc->memcpy(input2.ptr<float>(), ptr2, input2.byte_size(), MemcpyKind::kMemcpyCPU2CUDA);

    Tensor output(DataType::kDataTypeFp32, dims, true, alloc);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    VecAddLayer add_layer(DeviceType::kDeviceGPU);
    Layer& layer = add_layer;
    layer.set_cuda_config(std::make_shared<kernel::CudaConfig>(stream));

    ASSERT_EQ(layer.device_type(), DeviceType::kDeviceGPU);
    layer.forward(input1, input2, output);
    cudaDeviceSynchronize();

    ASSERT_EQ(input1.size(), 64);
    ASSERT_EQ(input1.dims_size(), 1);
    ASSERT_EQ(input2.size(), 64);
    ASSERT_EQ(input2.dims_size(), 1);
    ASSERT_EQ(output.size(), 64);
    ASSERT_EQ(output.dims_size(), 1);

    output.to_cpu();
    ASSERT_EQ(output.device_type(), DeviceType::kDeviceCPU);

    float* ptr = output.ptr<float>();
    for (int i = 0; i < output.size(); ++i) {
        ASSERT_FLOAT_EQ(ptr[i], 3.f);
    }

    delete[] ptr1;
    delete[] ptr2;
}