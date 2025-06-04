#include <glog/logging.h>

#include "op/embedding.h"
#include "kernel/kernels_interface.h"

namespace op {

EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t seq_len,
                               int32_t vocab_size, int32_t dim)
        : LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding"),
          seq_len_{seq_len}, vocab_size_{vocab_size}, dim_{dim} {
    reset_input_size(1);
    reset_weight_size(1);
    reset_output_size(1);
}

base::Status EmbeddingLayer::check() const {
    // 取张量
    auto input = get_input(0);
    auto output = get_output(0);
    auto weight = get_weight(0);

    base::Status status;

    // 检查 input
    status = check_tensor_with_dim(input, device_type_, base::DataType::kDataTypeInt32, seq_len_);
    if (!status) {
        LOG(ERROR) << "The input tensor error in the embedding layer.";
        return status;
    }

    // 检查 weight
    status = check_tensor_with_dim(weight, device_type_, data_type_, vocab_size_, dim_);
    if (!status) {
        LOG(ERROR) << "The weight tensor error in the embedding layer.";
        return status;
    }

    // 检查 output
    status = check_tensor_with_dim(output, device_type_, data_type_, seq_len_, dim_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the embedding layer.";
        return status;
    }

    return base::error::Success();
}

base::Status EmbeddingLayer::forward() {
    auto status = check();
    if (!status) {
        return status;
    }

    if (device_type_ == base::DeviceType::kDeviceGPU) {
        CHECK(cuda_config_ != nullptr);
    }

    // 取张量
    auto input = get_input(0);
    auto weight = get_weight(0);
    auto output = get_output(0);

    // 实际计算
    auto stream = cuda_config_ ? cuda_config_->stream : nullptr;
    auto accumulate_call = kernel::get_embedding_kernel(device_type_);
    accumulate_call(input, weight, output, stream);

    return base::error::Success();
}

} // namespace op