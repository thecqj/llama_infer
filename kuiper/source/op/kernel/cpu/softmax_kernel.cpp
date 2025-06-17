#include <glog/logging.h>
#include <armadillo>
#include <algorithm>

#include "softmax_kernel.h"

namespace kernel {

void softmax_inplace_cpu(const tensor::Tensor& input, void* stream) {
    const int32_t size = static_cast<int32_t>(input.size());
    const float* input_ptr = input.ptr<float>();

    // 求最大值
    float max_value = *std::max_element(input_ptr, input_ptr + size);

    // 求各元素的 exp(x - max_value)
    arma::fvec input_mat(const_cast<float*>(input_ptr), size, false, true);
    input_mat = arma::exp(input_mat - max_value);

    // 求 sum( exp(x - max_value) )
    float sum_value = arma::sum(input_mat);

    // 求 softmax 结果：exp(x - max_value) / sum( exp(x - max_value) )
    input_mat = input_mat / sum_value;
}

} // namespace kernel