#include <glog/logging.h>

#include "op/add.h"

namespace op {

VecAddLayer::VecAddLayer(base::DeviceType device_type) 
        : Layer(device_type, LayerType::kLayerAdd, "Add") {
    reset_input_size(2);
    reset_output_size(1);
}

base::Status VecAddLayer::check() const {
    // 取输入、输出张量
    auto input1 = get_input(0);
    auto input2 = get_input(1);
    auto output = get_output(0);

    // 计算张量大小，检查维度信息
    size_t size = input1.size();
    base::Status status;

    // 检查 input1
    status = check_tensor_with_dim(input1, device_type_, data_type_, size);
    if (!status) {
        LOG(ERROR) << "The input tensor 1 error in the add layer.";
        return status;
    }
    // 检查 input2
    status = check_tensor_with_dim(input2, device_type_, data_type_, size);
    if (!status) {
        LOG(ERROR) << "The input tensor 2 error in the add layer.";
        return status;
    }
    // 检查 output
    status = check_tensor_with_dim(output, device_type_, data_type_, size);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the add layer.";
        return status;
    }

    return base::error::Success();
}

base::Status VecAddLayer::forward() {
    // 检查张量信息
    auto status = check();
    if (!status) {
        return status;
    }

    // 取张量
    auto input1 = get_input(0);
    auto input2 = get_input(1);
    auto output = get_output(0);
    if (device_type_ == base::DeviceType::kDeviceGPU) {
        CHECK(cuda_config_ != nullptr);
    }

    // 实际计算
    LOG(INFO) << "DEBUG: run add_kernel()...";

    return base::error::Success();
}

} // namespace op