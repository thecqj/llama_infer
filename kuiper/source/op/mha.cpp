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
    reset_input_size(5);
    reset_output_size(1);
}

base::Status MultiHeadAttentionLayer::check() const {
    // 暂时不想做检查
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