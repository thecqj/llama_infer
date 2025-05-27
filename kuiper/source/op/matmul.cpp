#include <glog/logging.h>

#include "op/matmul.h"
#include "kernel/kernels_interface.h"

namespace op {

MatmulLayer::MatmulLayer(base::DeviceType device_type, base::DataType data_type, 
                         int32_t dim0, int32_t dim1, bool is_quant_layer, bool has_bias)
        : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul", data_type),
          dim0_{dim0}, dim1_{dim1}, has_bias_{has_bias} {
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
    if (has_bias_) {
        bias_.resize(1);
    }
}

base::Status MatmulLayer::check() const {
    // 取输入、输出、权重张量
    auto input = get_input(0);
    auto output = get_output(0);
    auto weight = get_weight(0);

    base::Status status;

    // 检查 input
    status = check_tensor_with_dim(input, device_type_, data_type_, dim1_);
    if (!status) {
        LOG(ERROR) << "The input tensor error in the Matmul layer.";
        return status;
    }
    // 检查 output
    status = check_tensor_with_dim(output, device_type_, data_type_, dim0_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the Matmul layer.";
        return status;
    }

    // 检查 weight
    if (!is_quant_layer_) {
        status = check_tensor_with_dim(weight, device_type_, base::DataType::kDataTypeFp32, dim0_, dim1_);
        if (!status) {
            LOG(ERROR) << "The weight tensor error in the Matmul layer.";
            return status;
        }
    } else {
        status = check_tensor_with_dim(weight, device_type_, base::DataType::kDataTypeInt8, dim0_, dim1_);
        if (!status) {
            LOG(ERROR) << "The weight tensor error in the Matmul layer.";
            return status;
        }
        // 还要检查量化系数
        int32_t scales_num = weight.size() / group_size_;
        status = check_tensor_with_dim(scales_, device_type_, base::DataType::kDataTypeFp32, scales_num);
        if (!status) {
            LOG(ERROR) << "The scale tensor error in the Matmul layer.";
            return status;
        }
    }

    return base::error::Success();
}

base::Status MatmulLayer::forward() {
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
    auto stream = cuda_config_ ? cuda_config_->stream : nullptr;
    if (!is_quant_layer_) {
        auto accumulate_call = kernel::get_matmul_kernel(device_type_);
        accumulate_call(input, weight, output, 1.f, stream);
    } else {
        auto accumulate_call = kernel::get_matmul_kernel_quant8(device_type_);
        accumulate_call(input, weight, output, group_size_, scales_, stream);
    }

    if (has_bias_) {
        auto bias = get_bias(0);
        auto accumulate_call = kernel::get_add_kernel(device_type_);
        accumulate_call(input, bias, output, stream);
    }

    return base::error::Success();
}

// -------------------------------- 偏置项相关函数 --------------------------------
base::Status MatmulLayer::set_bias(int32_t idx, const tensor::Tensor& bias) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    CHECK(bias.data_type() == data_type_);

    if (!bias.empty()) {
        CHECK(bias.device_type() == device_type_);
    }

    bias_[idx] = bias;
    return base::error::Success();
}

base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr, 
                                   base::DeviceType device_type) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    CHECK_NE(bias_ptr, nullptr);
    CHECK(device_type_ == device_type);

    if (!is_quant_layer_) {
        // 用权重数据指针创建非量化张量
        tensor::Tensor bias(base::DataType::kDataTypeFp32, dim, false, nullptr, 
                            const_cast<void*>(bias_ptr), device_type);
        bias_[idx] = bias;
    } else {
        // 用权重数据指针创建量化张量
        tensor::Tensor bias(base::DataType::kDataTypeInt8, dim, false, nullptr, 
                            const_cast<void*>(bias_ptr), device_type);
        bias_[idx] = bias;
        // 还需要设置量化信息
        const int32_t bias_size = static_cast<int32_t>(bias.size());
        CHECK(bias_size % group_size_ == 0);

        int32_t scale_num = bias_size / group_size_;
        scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_num, false, nullptr,
                                 reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size), device_type};
    }

    return base::error::Success();
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_[idx];
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_[idx];
}

void MatmulLayer::to_cuda() {
    LayerParam::to_cuda();
    if (has_bias_) {
        for (auto& bias : bias_) {
            bias.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
}

} // namespace op