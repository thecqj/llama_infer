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

// function
AddKernel get_add_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);
MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

EmbeddingKernel get_embedding_kernel(base::DeviceType device_type);


} // namespace kernel