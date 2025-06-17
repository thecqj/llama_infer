#pragma once

#include "tensor/tensor.h"

namespace kernel {

// typedef
using AddKernel = void(*)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          const tensor::Tensor& output, void* stream);

using RMSNormKernel = void(*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);

using MatmulKernel = void(*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale, void* stream);

using MatmulKernelQuant = void(*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, int32_t group_size,
                                  const tensor::Tensor& scale, void* stream);

using EmbeddingKernel = void(*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, void* stream);

using MHAKernel = void(*)(const int32_t layer_index, const int32_t pos,
                          const int32_t kv_mul, const int32_t kv_dim,
                          const int32_t seq_len, const int32_t head_num, const int32_t head_size,
                          const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                          const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                          const tensor::Tensor& output_tensor, void* stream);

// function
AddKernel get_add_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);
MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

EmbeddingKernel get_embedding_kernel(base::DeviceType device_type);

MHAKernel get_mha_kernel(base::DeviceType device_type);

} // namespace kernel