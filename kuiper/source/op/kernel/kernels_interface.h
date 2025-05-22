#pragma once

#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {

// typedef
using AddKernel = void(*)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          const tensor::Tensor& output, void* stream);

using RMSNormKernel = void(*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);

// function
AddKernel get_add_kernel(base::DeviceType device_type);
RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

} // namespace kernel