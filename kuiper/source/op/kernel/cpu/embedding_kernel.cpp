#include <glog/logging.h>
#include <armadillo>

#include "embedding_kernel.h"

namespace kernel {

void embedding_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, void* stream) {
    // 判空
    CHECK_EQ(input.empty(), false);
    CHECK_EQ(weight.empty(), false);
    CHECK_EQ(output.empty(), false);

    // 检查设备
    CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceCPU);
    const auto allocator = base::CPUDeviceAllocatorFactory::get_instance();

    // 检查大小
    const int32_t seq_len = static_cast<int32_t>(input.size());
    const int32_t vocab_size = weight.get_dim(0);
    const int32_t dim = weight.get_dim(1);

    CHECK_EQ(output.get_dim(0), seq_len);
    CHECK_EQ(output.get_dim(1), dim);

    // 取数据计算
    for (int32_t i = 0; i < seq_len; ++i) {
        int32_t token = input.index<int32_t>(i);
        if (token > vocab_size) {
            LOG(FATAL) << "Token index is greater than vocab size.";
        } else {
            float* src_ptr = const_cast<float*>(weight.ptr<float>(token * dim));
            float* dest_ptr = const_cast<float*>(output.ptr<float>(i * dim));
            allocator->memcpy(dest_ptr, src_ptr, dim * sizeof(float), base::MemcpyKind::kMemcpyCPU2CPU);
        }
    }

}

} // namespace kernel