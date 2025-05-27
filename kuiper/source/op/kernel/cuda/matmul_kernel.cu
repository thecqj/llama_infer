#include <glog/logging.h>

#include "matmul_kernel.cuh"

namespace kernel {

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale, void* stream) {

}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, void* stream) {

}

} // namespace kernel