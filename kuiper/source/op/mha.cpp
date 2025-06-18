#include <glog/logging.h>

#include "op/mha.h"
#include "kernel/kernels_interface.h"

namespace op {

MultiHeadAttentionLayer::MultiHeadAttentionLayer(base::DeviceType device_type, int32_t layer_index,
                                                 int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                                 int32_t head_num, int32_t head_size)
        : Layer(device_type, LayerType::kLayerMHA, "MultiHeadAttenstion"),
          layer_index_{layer_index}, kv_mul_{kv_mul}, kv_dim_{kv_dim},
          seq_len_{seq_len}, head_num_{head_num}, head_size_{head_size} {
    reset_input_size(4);
    reset_output_size(1);
}

base::Status MultiHeadAttentionLayer::check() const {
    // 取张量
    auto& query_tensor = get_input(0);
    auto& score_tensor = get_input(1);
    auto& key_cache_tensor = get_input(2);
    auto& value_cache_tensor = get_input(3);
    auto& output_tensor = get_output(0);

    base::Status status;

    // 检查 query_tensor
    status = check_tensor_with_dim(query_tensor, device_type_, data_type_, head_num_, head_size_);
    if (!status) {
        LOG(ERROR) << "The query_tensor error in the matmul layer.";
        return status;
    }

    // 检查 score_tensor
    status = check_tensor_with_dim(score_tensor, device_type_, data_type_, head_num_, seq_len_);
    if (!status) {
        LOG(ERROR) << "The score_tensor error in the matmul layer.";
        return status;
    }

    // 检查 key_cache_tensor
    status = check_tensor_with_dim(key_cache_tensor, device_type_, data_type_,
                                   pos_ + 1, head_num_ / kv_mul_, head_size_);
    if (!status) {
        LOG(ERROR) << "The key_cache_tensor error in the matmul layer.";
        return status;
    }

    // 检查 value_cache_tensor
    status = check_tensor_with_dim(value_cache_tensor, device_type_, data_type_,
                                   pos_ + 1, head_num_ / kv_mul_, head_size_);
    if (!status) {
        LOG(ERROR) << "The value_cache_tensor error in the matmul layer.";
        return status;
    }

    // 检查 output
    status = check_tensor_with_dim(output_tensor, device_type_, data_type_, head_num_, head_size_);
    if (!status) {
        LOG(ERROR) << "The output_tensor error in the matmul layer.";
        return status;
    }

    return base::error::Success();
}

base::Status MultiHeadAttentionLayer::forward() {
    auto status = check();
    if (!status) {
        return status;
    }

    if (device_type_ == base::DeviceType::kDeviceGPU) {
        CHECK(cuda_config_ != nullptr);
    }

    // 取张量
    const tensor::Tensor& query_tensor = get_input(0);
    const tensor::Tensor& score_tensor = get_input(1);
    const tensor::Tensor& key_cache_tensor = get_input(2);
    const tensor::Tensor& value_cache_tensor = get_input(3);
    const tensor::Tensor& output_tensor = get_output(0);

    // 实际计算
    auto stream = cuda_config_ ? cuda_config_->stream : nullptr;
    auto accumulate_call = kernel::get_mha_kernel(device_type_);
    accumulate_call(layer_index_, pos_, kv_mul_, kv_dim_, seq_len_, head_num_, head_size_,
                    query_tensor, score_tensor, key_cache_tensor, value_cache_tensor, output_tensor,
                    stream);

    return base::error::Success();
}

} // namespace op