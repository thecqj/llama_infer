#include <glog/logging.h>

#include "add_kernel.cuh"

namespace kernel {

__global__ void add_kernel_cu_fp32(const float* input1, const float* input2,
                                  float* output, const size_t size) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    float val1 = input1[tid];
    float val2 = input2[tid];
    output[tid] = val1 + val2;
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream) {
    // 判空
    CHECK_EQ(input1.empty(), false);
    CHECK_EQ(input2.empty(), false);
    CHECK_EQ(output.empty(), false);

    // 检查大小
    const size_t size = input1.size();
    CHECK_EQ(input2.size(), size);
    CHECK_EQ(output.size(), size);

    // 设置网格大小
    const int BLOCK_SIZE = 512;
    dim3 block(BLOCK_SIZE);
    dim3 grid((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 启动核函数
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        add_kernel_cu_fp32<<<grid, block, 0, stream_>>>(
            input1.ptr<float>(), input2.ptr<float>(),
            const_cast<float*>(output.ptr<float>()), size);
    } else {
        add_kernel_cu_fp32<<<grid, block>>>(
            input1.ptr<float>(), input2.ptr<float>(),
            const_cast<float*>(output.ptr<float>()), size);
    }
}

} // namespace kernel