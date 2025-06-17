#pragma once

#include "tensor/tensor.h"

namespace kernel {

void mha_kernel_cu(const int32_t layer_index, const int32_t pos,
                   const int32_t kv_mul, const int32_t kv_dim,
                   const int32_t seq_len, const int32_t head_num, const int32_t head_size,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   const tensor::Tensor& output_tensor, void* stream = nullptr);

} // namespace kernel