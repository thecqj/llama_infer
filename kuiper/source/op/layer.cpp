#include <glog/logging.h>
#include <numeric>

#include "op/layer.h"

namespace op {

// ---------------------------------- 无参算子类 ----------------------------------
// 输入输出有关函数定义
inline void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    inputs_[idx] = input;
}

inline void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    outputs_[idx] = output;
}

inline tensor::Tensor& Layer::get_input(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_[idx];
}

inline tensor::Tensor& Layer::get_output(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_[idx];
}

inline const tensor::Tensor& Layer::get_input(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_[idx];
}

inline const tensor::Tensor& Layer::get_output(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_[idx];
}

// 检查相关函数定义
base::Status Layer::check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                                 base::DataType data_type) const {
    if (tensor.empty()) {
        return base::error::InvalidArgument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type) {
        return base::error::InvalidArgument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type) {
        return base::error::InvalidArgument("The tensor has a wrong data type.");
    }
    return base::error::Success();
}

// 设备相关函数定义
void Layer::to_cuda() {
    // 修改层的设备类型
    device_type_ = base::DeviceType::kDeviceGPU;
    // 将输入张量移动到cuda
    for (auto& tensor : inputs_) {
        tensor.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
    // 将输出张量移动到cuda
    for (auto& tensor : outputs_) {
        tensor.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
}

// forward 函数组定义
base::Status Layer::forward(const tensor::Tensor& input, const tensor::Tensor& output) {
    // 重置大小
    reset_input_size(1);
    reset_output_size(1);
    // 设置张量
    set_input(0, input);
    set_output(0, output);
    // 计算
    return forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output) {
    reset_input_size(2);
    reset_output_size(1);

    set_input(0, input1);
    set_input(1, input2);
    
    set_output(0, output);

    return forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output) {
    reset_input_size(3);
    reset_output_size(1);
    
    set_input(0, input1);
    set_input(1, input2);
    set_input(2, input3);

    set_output(0, output);

    return forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& output) {
    reset_input_size(4);
    reset_output_size(1);
                                
    set_input(0, input1);
    set_input(1, input2);
    set_input(2, input3);
    set_input(3, input4);

    set_output(0, output);

    return forward();
}

base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& input5, const tensor::Tensor& output) {
    reset_input_size(5);
    reset_output_size(1);
                                
    set_input(0, input1);
    set_input(1, input2);
    set_input(2, input3);
    set_input(3, input4);
    set_input(4, input5);

    set_output(0, output);

    return forward();
}

// ---------------------------------- 带参算子类 ----------------------------------
// 权重相关函数定义
tensor::Tensor& LayerParam::get_weight(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_[idx];
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_[idx];
}

base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    CHECK(weight.data_type() == data_type_);

    // 不是空张量，还需检查设备类型是否一致
    if (!weight.empty()) {
        CHECK(weight.device_type() == device_type_);
    }

    weights_[idx] = weight;
    return base::error::Success();
}

// 使用指针创建张量，并设置权重
base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, base::DeviceType device_type) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    CHECK_NE(weight_ptr, nullptr);
    CHECK(device_type_ == device_type);

    // 用权重数据指针创建张量
    tensor::Tensor weight(data_type_, dims, false, nullptr, const_cast<void*>(weight_ptr), device_type);
    weights_[idx] = weight;

    return base::error::Success();
}

// 设备相关
void LayerParam::to_cuda() {
    Layer::to_cuda();
    for (auto& weight: weights_) {
        weight.to_cuda();
    }
}

} // namespace op