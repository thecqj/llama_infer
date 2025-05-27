#pragma once

#include "tensor/tensor.h"

namespace kernel {

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      void* stream = nullptr);

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, void* stream = nullptr);

} // namespace kernel