#include <glog/logging.h>

#include "embedding_kernel.cuh"

namespace kernel {

void embedding_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                         const tensor::Tensor& output, void* stream) {

}

} // namespace kernel