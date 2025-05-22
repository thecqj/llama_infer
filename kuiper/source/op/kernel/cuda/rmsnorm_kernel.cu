#include <glog/logging.h>
#include <cub/block/block_reduce.cuh>
#include <vector_types.h>

#include "rmsnorm_kernel.cuh"

namespace kernel {

template <const int BLOCK_SIZE>
static __global__ void row_rmsnorm_f32(float* input, float* weight, 
                                       float* output, int size, float eps) {
    const int tid = threadIdx.x;

    constexpr int pack_size = 4;
    const int pack_num = size / pack_size;
    const int pack_off = pack_num * pack_size;  // 剩余元素的起始索引

    // 求每个线程处理部分平方和
    float sum = 0.f;
    float4* input_pack = reinterpret_cast<float4*>(input);
    for (int i = tid; i < pack_num; i += BLOCK_SIZE) {  // 4个一组平方和
        float4 reg_x = *(input_pack + i);
        sum += reg_x.x * reg_x.x + reg_x.y * reg_x.y +
               reg_x.z * reg_x.z + reg_x.w * reg_x.w;
    }
    const int pack_off_idx = pack_off + tid;
    sum += pack_off_idx < size ? input[pack_off_idx] * input[pack_off_idx] : 0.f; // 剩余元素的平方和

    // 规约求输入的平方和：sum(x ^ 2)
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp;  // 为 BlockReduce 分配的共享内存
    __shared__ float shared_val;    // 用来存储规约结果，所有线程共享

    sum = BlockReduce(temp).Sum(sum);   // 规约求和
    if (tid == 0) shared_val = sum; // 第一个线程将结果写入共享内存中
    __syncthreads();

    // rsqrt = 1 / sqrt(sum / d + eps)
    sum = shared_val;
    const float rsqrt = rsqrtf(sum / static_cast<float>(size) + eps);

    // 归一化结果写入 output：y = x * rsqrt * w
    float4* weight_pack = reinterpret_cast<float4*>(weight);
    float4* output_pack = reinterpret_cast<float4*>(output);
    for (int i = tid; i < pack_num; i += BLOCK_SIZE) {  // 四个一组写入
        float4 reg_x = *(input_pack + i);
        float4 reg_w = *(weight_pack + i);
        *(output_pack + i) = make_float4(rsqrt * reg_x.x * reg_w.x, rsqrt * reg_x.y * reg_w.y,
                                         rsqrt * reg_x.z * reg_w.z, rsqrt * reg_x.w * reg_w.w);
    }
    if (pack_off_idx < size) {  // 剩余部分写入
        output[pack_off_idx] = rsqrt * input[pack_off_idx] * weight[pack_off_idx];
    }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
    // 判空
    CHECK_EQ(input.empty(), false);
    CHECK_EQ(weight.empty(), false);
    CHECK_EQ(output.empty(), false);

    // 检查设备类型
    CHECK(input.device_type() == base::DeviceType::kDeviceGPU &&
          weight.device_type() == base::DeviceType::kDeviceGPU &&
          output.device_type() == base::DeviceType::kDeviceGPU);

    // 检查大小
    CHECK_EQ(input.size(), weight.size());
    CHECK_EQ(input.size(), output.size());

#ifdef QWEN2_SUPPORT
    const float eps = 1e-6f;
#else
    const float eps = 1e-5f;
#endif

    // 设置网格大小
    const int size = input.size();
    const int BLOCK_SIZE = 128;
    dim3 block(BLOCK_SIZE); // 128个线程处理一行
    dim3 grid(1);   // 行级RMSNorm，因此只有一个block
    
    // 启动内核
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        row_rmsnorm_f32<BLOCK_SIZE><<<grid, block, 0, stream_>>>(
            const_cast<float*>(input.ptr<float>()), const_cast<float*>(weight.ptr<float>()),
            const_cast<float*>(output.ptr<float>()), size, eps);
    } else {
        row_rmsnorm_f32<BLOCK_SIZE><<<grid, block>>>(
            const_cast<float*>(input.ptr<float>()), const_cast<float*>(weight.ptr<float>()),
            const_cast<float*>(output.ptr<float>()), size, eps);
    }

}

} // namespace kernel