#include <glog/logging.h>
#include <armadillo>

#include "matmul_kernel.h"

namespace kernel {

void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale, void* stream) {
    // 判空
    CHECK_EQ(input.empty(), false);
    CHECK_EQ(weight.empty(), false);
    CHECK_EQ(output.empty(), false);

    // 检查设备
    CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

    // 检查大小
    CHECK_EQ(weight.dims_size(), 2);
    const int32_t weight_dim0 = weight.get_dim(0);
    const int32_t weight_dim1 = weight.get_dim(1);

    const int32_t input_dim = input.size();
    CHECK_EQ(input_dim, weight_dim1);

    const int32_t output_dim = output.size();
    CHECK_EQ(output_dim, weight_dim0);

    // 取数据计算
    auto data_type = input.data_type();
    if (data_type == base::DataType::kDataTypeFp32) {
        arma::fmat input_mat(const_cast<float*>(input.ptr<float>()), 1, input_dim, false, true);
        arma::fmat weight_mat(const_cast<float*>(weight.ptr<float>()), weight_dim1, weight_dim0, false, true);
        arma::fmat output_mat(const_cast<float*>(output.ptr<float>()), 1, output_dim, false, true);

        output_mat = (input_mat * weight_mat) * scale;
    }
}

} // namespace kernel