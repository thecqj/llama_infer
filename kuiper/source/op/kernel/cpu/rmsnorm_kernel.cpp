#include <glog/logging.h>
#include <armadillo>

#include "rmsnorm_kernel.h"

namespace kernel {

// input:  1 x hidden_dim
// weight: 1 x hidden_dim
// output: 1 x hidden_dim
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
    // 判空
    CHECK_EQ(input.empty(), false);
    CHECK_EQ(weight.empty(), false);
    CHECK_EQ(output.empty(), false);

    // 检查设备
    CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
          weight.device_type() == base::DeviceType::kDeviceCPU &&
          output.device_type() == base::DeviceType::kDeviceCPU);

    // 检查大小
    CHECK_EQ(input.size(), weight.size());
    CHECK_EQ(input.size(), output.size());
    
    // 取数据
    arma::fvec input_vec(const_cast<float*>(input.ptr<float>()), input.size(), false, true);
    arma::fvec weight_vec(const_cast<float*>(weight.ptr<float>()), weight.size(), false, true);
    arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);

#ifdef QWEN2_SUPPORT
    const float eps = 1e-6f;
#else
    const float eps = 1e-5f;
#endif

    // mean = sum(x^2) / d + eps
    const float mean = arma::as_scalar(arma::mean(arma::pow(input_vec, 2))) + eps;
    // rsqrt = 1 / sqrt(mean)
    const float rsqrt = 1.f / std::sqrt(mean);
    // y = x * rsqrt * w
    output_vec = weight_vec % (rsqrt * input_vec);  // 重载%，逐元素相乘
}

} // namespace kernel