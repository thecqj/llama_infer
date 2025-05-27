#include <glog/logging.h>

#include "op/rmsnorm.h"
#include "kernel/kernels_interface.h"

namespace op {

RMSNormLayer::RMSNormLayer(base::DeviceType device_type, base::DataType data_type)
        : LayerParam(device_type, LayerType::kLayerRMSNorm, false, "RMSNorm", data_type) {
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
}

base::Status RMSNormLayer::check() const {
    // 取输入、输出、权重张量
    auto input = get_input(0);
    auto output = get_output(0);
    auto weight = get_weight(0);
    // 取 input 的大小（都是一维向量）
    int32_t size = input.size();

    base::Status status;

    // 检查 input
    status = check_tensor(input, device_type_, data_type_);
    if (!status) {
        LOG(ERROR) << "The input tensor error in the RMSNorm layer.";
        return status;
    }
    // 检查 output
    status = check_tensor_with_dim(output, device_type_, data_type_, size);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the RMSNorm layer.";
        return status;
    }
    // 检查 weight
    status = check_tensor_with_dim(weight, device_type_, data_type_, size);
    if (!status) {
        LOG(ERROR) << "The weight tensor error in the RMSNorm layer.";
        return status;
    }

    return base::error::Success();
}

base::Status RMSNormLayer::forward() {
    // 检查张量信息
    auto status = check();
    if (!status) {
        return status;
    }

    if (device_type_ == base::DeviceType::kDeviceGPU) {
        CHECK(cuda_config_ != nullptr);
    }

    // 取张量
    auto input = get_input(0);
    auto output = get_output(0);
    auto weight = get_weight(0);

    // 实际计算
    auto accumulate_call = kernel::get_rmsnorm_kernel(device_type_);
    accumulate_call(input, weight, output, cuda_config_ ? cuda_config_->stream : nullptr);

    return base::error::Success();
}

} // namespace op