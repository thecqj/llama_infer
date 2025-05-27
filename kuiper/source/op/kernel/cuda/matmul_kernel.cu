#include <glog/logging.h>
#include <vector_types.h>
#include <cub/block/block_reduce.cuh>

#include "matmul_kernel.cuh"

namespace kernel {

// GEMV: output = weight * input
//  input:  M x 1
//  weight: K x M
//  output: K x 1
template <const int BLOCK_SIZE = 128>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output,
                                      int K, int M) {
    const int tid = threadIdx.x;
    const int bx = blockIdx.x;  // 第几个block，即第几行

    constexpr int pack_size = 4;
    const int pack_num = M / pack_size;
    const int pack_off = pack_num * pack_size;

    // 求每个线程处理部分乘积和
    float sum = 0.f;
    float4* input_pack = reinterpret_cast<float4*>(const_cast<float*>(input));
    float4* weight_pack = reinterpret_cast<float4*>(const_cast<float*>(weight + bx * M));
    #pragma unroll
    for (int i = tid; i < pack_num; i += BLOCK_SIZE) {
        float4 reg_x = input_pack[i];
        float4 reg_w = weight_pack[i];
        sum += reg_x.x * reg_w.x + reg_x.y * reg_w.y +
               reg_x.z * reg_w.z + reg_x.w * reg_w.w;
    }
    const int pack_off_idx = pack_off + tid;
    sum += pack_off_idx < M ? input[pack_off_idx] * weight[bx * M + pack_off_idx] : 0.f;

    // 规约求和
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp;  // 为 BlockReduce 分配的共享内存

    sum = BlockReduce(temp).Sum(sum);   // 规约求和
    __syncthreads();

    // 将结果写到output
    if (tid == 0) {
        output[bx] = sum;
    }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale, void* stream) {
    // 判空
    CHECK_EQ(input.empty(), false);
    CHECK_EQ(weight.empty(), false);
    CHECK_EQ(output.empty(), false);

    // 检查设备类型
    CHECK(input.device_type() == base::DeviceType::kDeviceGPU);
    CHECK(weight.device_type() == base::DeviceType::kDeviceGPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceGPU);

    // 检查大小
    CHECK_EQ(weight.dims_size(), 2);
    const int32_t weight_dim0 = weight.get_dim(0);
    const int32_t weight_dim1 = weight.get_dim(1);

    const int32_t input_dim = input.size();
    CHECK_EQ(input_dim, weight_dim1);

    const int32_t output_dim = output.size();
    CHECK_EQ(output_dim, weight_dim0);

    // 设置网格大小
    const int BLOCK_SIZE = 128;
    dim3 block(BLOCK_SIZE); // 128个线程处理一行
    dim3 grid(weight_dim0); // 共 dim0 行，即 dim0 个 block

    // 启动内核
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        matmul_kernel_cu_fp32<128><<<grid, block, 0, stream_>>>(
            input.ptr<float>(), weight.ptr<float>(),
            const_cast<float*>(output.ptr<float>()),
            weight_dim0, weight_dim1);
    } else {
        matmul_kernel_cu_fp32<128><<<grid, block>>>(
            input.ptr<float>(), weight.ptr<float>(),
            const_cast<float*>(output.ptr<float>()),
            weight_dim0, weight_dim1);
    }
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, void* stream) {

}

} // namespace kernel