#pragma once

#include "tensor/tensor.h"

namespace kernel {

void scale_sum_kernel_cpu(const tensor::Tensor& value, const tensor::Tensor& scale,
                          const tensor::Tensor& output, const int pos, const int size,
                          const int stride, void* stream = nullptr);

} // namespace kernel