#include <glog/logging.h>

#include "op/add.h"
#include "kernel/kernels_interface.h"

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

    base::Status status;

    // 检查 input1
    status = check_tensor(input1, device_type_, data_type_);
    if (!status) {
        LOG(ERROR) << "The input tensor 1 error in the add layer.";
        return status;
    }
    // 检查 input2
    status = check_tensor(input2, device_type_, data_type_);
    if (!status) {
        LOG(ERROR) << "The input tensor 2 error in the add layer.";
        return status;
    }
    // 检查 output
    status = check_tensor(output, device_type_, data_type_);
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
    auto accumulate_call = kernel::get_add_kernel(device_type_);
    accumulate_call(input1, input2, output, cuda_config_ ? cuda_config_->stream : nullptr);

    return base::error::Success();
}

} // namespace op