#include <glog/logging.h>

#include "embedding_kernel.cuh"

namespace kernel {

__global__ void embedding_kernel_cu_fp32(const int* input, const float* weight, float* output,
                                         const int32_t seq_len, const int32_t vocab_size,
                                         const int32_t dim) {
    const int bid = blockIdx.x;     // 处理第几个token
    if (bid >= seq_len) return;     // 超过序列长度
    
    const int token = input[bid];
    if (token >= vocab_size) return;    // 超过词汇表大小

    const float* weight_start = weight + token * dim;
    float* output_start = output + bid * dim;

    for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        output_start[i] = weight_start[i];
    }
}

void embedding_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                         const tensor::Tensor& output, void* stream) {
    // 判空
    CHECK_EQ(input.empty(), false);
    CHECK_EQ(weight.empty(), false);
    CHECK_EQ(output.empty(), false);

    // 检查设备
    CHECK(input.device_type() == base::DeviceType::kDeviceGPU);
    CHECK(weight.device_type() == base::DeviceType::kDeviceGPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceGPU);

    // 检查大小
    const int32_t seq_len = static_cast<int32_t>(input.size());
    const int32_t vocab_size = weight.get_dim(0);
    const int32_t dim = weight.get_dim(1);

    CHECK_EQ(output.get_dim(0), seq_len);
    CHECK_EQ(output.get_dim(1), dim);

    // 设置网格大小
    const int BLOCK_SIZE = 128;
    dim3 block(BLOCK_SIZE);     // 每个线程块处理一个token
    dim3 grid(seq_len);

    // 启动内核
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        embedding_kernel_cu_fp32<<<grid, block, 0, stream_>>>(
            input.ptr<int>(), weight.ptr<float>(),
            const_cast<float*>(output.ptr<float>()),
            seq_len, vocab_size, dim);
    } else {
        embedding_kernel_cu_fp32<<<grid, block>>>(
            input.ptr<int>(), weight.ptr<float>(),
            const_cast<float*>(output.ptr<float>()),
            seq_len, vocab_size, dim);
    }
}

} // namespace kernel