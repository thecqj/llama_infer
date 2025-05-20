#include <glog/logging.h>
#include <gtest/gtest.h>

#include "op/layer.h"

TEST(test_layer, test_io) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    Tensor input1(DataType::kDataTypeFp32, 64, true, alloc);
    Tensor input2(DataType::kDataTypeFp32, 64, true, alloc);
    Tensor output(DataType::kDataTypeFp32, 64, true, alloc);

    Layer layer(DeviceType::kDeviceCPU, LayerType::kLayerUnknown);
    layer.reset_input_size(2);
    layer.reset_output_size(1);
    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    ASSERT_EQ(layer.input_size(), 2);
    ASSERT_EQ(layer.output_size(), 1);

    layer.reset_input_size(1);
    layer.reset_output_size(2);
    ASSERT_EQ(layer.input_size(), 1);
    ASSERT_EQ(layer.output_size(), 2);

    auto t1 = layer.get_input(0);
    auto t2 = layer.get_output(0);
    ASSERT_EQ(t1.size(), input1.size());
    ASSERT_EQ(t2.size(), output.size());
}

TEST(test_layer, test_cuda_config) {
    using namespace base;
    using namespace op;

    auto config = std::make_shared<kernel::CudaConfig>();

    Layer layer(DeviceType::kDeviceCPU, LayerType::kLayerUnknown);
    layer.set_cuda_config(config);

    ASSERT_EQ(layer.cuda_config()->stream, nullptr);
}

TEST(test_layer, test_to_cuda) {
    using namespace base;
    using namespace tensor;
    using namespace op;
    
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    Tensor input1(DataType::kDataTypeFp32, 64, true, alloc);
    Tensor input2(DataType::kDataTypeFp32, 64, true, alloc);
    Tensor output(DataType::kDataTypeFp32, 64, true, alloc);

    Layer layer(DeviceType::kDeviceCPU, LayerType::kLayerUnknown);
    layer.reset_input_size(2);
    layer.reset_output_size(1);
    layer.set_input(0, input1);
    layer.set_input(1, input2);
    layer.set_output(0, output);

    layer.to_cuda();

    input1 = layer.get_input(0);
    input2 = layer.get_input(1);
    output = layer.get_output(0);

    ASSERT_EQ(layer.device_type(), DeviceType::kDeviceGPU);
    ASSERT_EQ(input1.device_type(), DeviceType::kDeviceGPU);
    ASSERT_EQ(input2.device_type(), DeviceType::kDeviceGPU);
    ASSERT_EQ(output.device_type(), DeviceType::kDeviceGPU);
}

TEST(test_layer, test_init) {
    using namespace base;
    using namespace op;

    Layer layer(DeviceType::kDeviceCPU, LayerType::kLayerUnknown);
    auto status = layer.init();

    ASSERT_EQ(status.get_err_code(), StatusCode::kSuccess);
}

TEST(test_layer, test_check) {
    using namespace base;
    using namespace op;

    Layer layer(DeviceType::kDeviceCPU, LayerType::kLayerUnknown);
    auto status = layer.check();

    ASSERT_EQ(status.get_err_code(), StatusCode::kFunctionUnImplement);
}

TEST(test_layer, test_forward) {
    using namespace base;
    using namespace op;
    using namespace tensor;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    Tensor input1(DataType::kDataTypeFp32, 64, true, alloc);
    Tensor input2(DataType::kDataTypeFp32, 64, true, alloc);
    Tensor input3(DataType::kDataTypeFp32, 64, true, alloc);
    Tensor input4(DataType::kDataTypeFp32, 64, true, alloc);
    Tensor input5(DataType::kDataTypeFp32, 64, true, alloc);
    Tensor output(DataType::kDataTypeFp32, 64, true, alloc);

    Layer layer(DeviceType::kDeviceCPU, LayerType::kLayerUnknown);

    auto status = layer.forward();
    ASSERT_EQ(status.get_err_code(), StatusCode::kFunctionUnImplement);

    status = layer.forward(input1, output);
    ASSERT_EQ(status.get_err_code(), StatusCode::kFunctionUnImplement);

    status = layer.forward(input1, input2, output);
    ASSERT_EQ(status.get_err_code(), StatusCode::kFunctionUnImplement);

    status = layer.forward(input1, input2, input3, output);
    ASSERT_EQ(status.get_err_code(), StatusCode::kFunctionUnImplement);

    status = layer.forward(input1, input2, input3, input4, output);
    ASSERT_EQ(status.get_err_code(), StatusCode::kFunctionUnImplement);

    status = layer.forward(input1, input2, input3, input4, input5, output);
    ASSERT_EQ(status.get_err_code(), StatusCode::kFunctionUnImplement);
}

TEST(test_layer, test_check_tensor) {
    using namespace base;
    using namespace op;
    using namespace tensor;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    Layer layer(DeviceType::kDeviceCPU, LayerType::kLayerUnknown);
    Status status;

    Tensor tensor1;
    status = layer.check_tensor(tensor1, DeviceType::kDeviceCPU, DataType::kDataTypeFp32);
    ASSERT_EQ(status.get_err_code(), StatusCode::kInvalidArgument);

    Tensor tensor2(DataType::kDataTypeFp32, 64, true, alloc);
    status = layer.check_tensor(tensor2, DeviceType::kDeviceGPU, DataType::kDataTypeFp32);
    ASSERT_EQ(status.get_err_code(), StatusCode::kInvalidArgument);

    status = layer.check_tensor(tensor2, DeviceType::kDeviceCPU, DataType::kDataTypeInt32);
    ASSERT_EQ(status.get_err_code(), StatusCode::kInvalidArgument);

    status = layer.check_tensor(tensor2, DeviceType::kDeviceCPU, DataType::kDataTypeFp32);
    ASSERT_EQ(status.get_err_code(), StatusCode::kSuccess);
}

TEST(test_layer, test_check_tensor_with_dim) {
    using namespace base;
    using namespace op;
    using namespace tensor;
    auto alloc = CUDADeviceAllocatorFactory::get_instance();

    std::vector<int32_t> dims{1, 2, 2, 4};
    Tensor tensor(DataType::kDataTypeFp32, dims, true, alloc);

    Layer layer(DeviceType::kDeviceGPU, LayerType::kLayerUnknown);
    Status status;

    status = layer.check_tensor_with_dim(tensor, DeviceType::kDeviceGPU, 
                                         DataType::kDataTypeFp32, 4);
    ASSERT_EQ(status.get_err_code(), StatusCode::kInvalidArgument);

    status = layer.check_tensor_with_dim(tensor, DeviceType::kDeviceGPU, 
                                         DataType::kDataTypeFp32, 1, 2, 2, 4, 5);
    ASSERT_EQ(status.get_err_code(), StatusCode::kInvalidArgument);

    status = layer.check_tensor_with_dim(tensor, DeviceType::kDeviceGPU, 
                                         DataType::kDataTypeFp32, 1, 2, 3, 4);
    ASSERT_EQ(status.get_err_code(), StatusCode::kInvalidArgument);

    status = layer.check_tensor_with_dim(tensor, DeviceType::kDeviceGPU, 
                                         DataType::kDataTypeFp32, 1, 2, 2, 4);
    ASSERT_EQ(status.get_err_code(), StatusCode::kSuccess);

}

TEST(test_layer, test_weight) {
    using namespace base;
    using namespace op;
    using namespace tensor;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    std::vector<int32_t> dims{1, 2, 2, 5};

    Tensor weight1(DataType::kDataTypeFp32, dims, true, alloc);
    float* ptr = new float[20];

    LayerParam layer(DeviceType::kDeviceCPU, LayerType::kLayerUnknown);
    layer.reset_weight_size(2);
    layer.set_weight(0, weight1);
    layer.set_weight(1, dims, ptr, DeviceType::kDeviceCPU);
    
    ASSERT_EQ(layer.weight_size(), 2);
    ASSERT_EQ(layer.get_weight(0).size(), 20);
    ASSERT_EQ(layer.get_weight(1).dims_size(), 4);

    layer.to_cuda();
    ASSERT_EQ(layer.device_type(), DeviceType::kDeviceGPU);
    ASSERT_EQ(layer.get_weight(0).device_type(), DeviceType::kDeviceGPU);
    ASSERT_EQ(layer.get_weight(1).device_type(), DeviceType::kDeviceGPU);
}