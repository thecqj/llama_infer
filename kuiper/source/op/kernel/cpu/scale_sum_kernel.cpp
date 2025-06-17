#include <glog/logging.h>
#include <armadillo>

#include "scale_sum_kernel.h"

namespace kernel {

void scale_sum_kernel_cpu(const tensor::Tensor& value, const tensor::Tensor& scale,
                          const tensor::Tensor& output, const int pos, const int size,
                          const int stride, void* stream) {
    CHECK_EQ(value.empty(), false);
    CHECK_EQ(scale.empty(), false);
    CHECK_EQ(output.empty(), false);
    CHECK_EQ(size, output.size());
    CHECK_EQ((pos + 1) * size, value.size());

    arma::fvec scale_vec(const_cast<float*>(scale.ptr<float>()), scale.size(), false, true);
    arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);

    for (int i = 0; i <= pos; ++i) {
        arma::fvec value_vec(const_cast<float*>(value.ptr<float>()) + i * stride, size, false, true);
        output_vec += scale_vec[i] * value_vec;
    }
}

} // namespace kernel