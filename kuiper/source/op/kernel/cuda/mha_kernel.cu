#include <glog/logging.h>
#include <cfloat>
#include <vector_types.h>
#include <cub/block/block_reduce.cuh>

#include "mha_kernel.cuh"

namespace kernel {

template <const int BLOCK_SIZE = 256>
__device__ void softmax_cu(float* __restrict__ x, int size) {
    const int tid = threadIdx.x;
    const int step = BLOCK_SIZE;

    // 每个线程求局部最大值
    float max_val = tid < size ? x[tid] : FLT_MIN;
    for (int i = tid + step; i < size; i += step) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // 规约求全局最大值，并写入共享内存
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (tid == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    // x = exp(x - max_val)，并求局部和
    float sum_val = 0.f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum_val += x[i];
    }

    // 规约求全局和，并写入共享内存
    sum_val = BlockReduce(temp).Sum(sum_val);
    if (tid == 0) {
        shared_val = sum_val;
    }
    __syncthreads();
    sum_val = shared_val;

    // x = exp(x - max_val) / sum_val
    for (int i = tid; i < size; i += step) {
        x[i] /= sum_val;
    }
}

__global__ void multi_head_attention_kernel(float* query, float* score, float* output,
                                            float* key_cache, float* value_cache,
                                            const int32_t pos, const int32_t kv_mul,
                                            const int32_t kv_dim, const int32_t seq_len,
                                            const int32_t head_num, const int32_t head_size,
                                            const int32_t layer_offset) {
    const float scale = 1.f / sqrtf(float(head_size));

    const int32_t head_idx = blockIdx.x;    // 第几个头
    if (head_idx >= head_num) {
        return;
    }

    extern __shared__ float s_query_head[]; // 共享内存存储 query 向量
    float* query_head = query + head_idx * head_size;   // 取该头下的 query 向量
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        // 加载 query 向量到共享内存
        s_query_head[i] = query_head[i];
    }
    __syncthreads();

    float* score_head = score + head_idx * seq_len; // 取该头下的 score 向量
    const int kv_head_offset = (head_idx / kv_mul) * head_size; // 该层下kv矩阵的头偏移量

    // 计算自注意力分数
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {  // 一个线程对应k的一行
        float* key_head = key_cache + layer_offset + t * kv_dim + kv_head_offset;   // 取该层该头的key向量
        
        float score_result = 0.f; // 得分
        for (int i = 0; i < head_size; i += 4) {    // 向量化存储
            float4 query_reg = *reinterpret_cast<float4*>(s_query_head + i);
            float4 key_reg = *reinterpret_cast<float4*>(key_head + i);

            score_result += query_reg.x * key_reg.x + query_reg.y * key_reg.y +
                            query_reg.z * key_reg.z + query_reg.w * key_reg.w;
        }

        score_result *= scale;  // 做缩放
        score_head[t] = score_result;   // 写入结果
    }
    __syncthreads();    // 必须等自注意力分数计算完毕后再执行后续计算

    // 做 softmax
    softmax_cu(score_head, pos + 1);
    __syncthreads();    // 同理，需要同步

    float* output_head = output + head_idx * head_size; // 取该头下的 output 向量

    // 使用自注意力分数对 value 矩阵加权
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) { // 一个线程对应 value 的一列
        float value_result = 0.f;
        for (int t = 0; t <= pos; ++t) {    // 向量点积
            float* value_head = value_cache + layer_offset + t * kv_dim + kv_head_offset;   // 第 t 行
            value_result += score_head[t] * value_head[i];
        }
        output_head[i] = value_result;
    }
}

void mha_kernel_cu(const int32_t layer_index, const int32_t pos,
                   const int32_t kv_mul, const int32_t kv_dim,
                   const int32_t seq_len, const int32_t head_num, const int32_t head_size,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   const tensor::Tensor& output_tensor, void* stream) {
    // 计算层偏移
    const int32_t layer_offset = layer_index * seq_len * kv_dim;

    // 取底层指针
    float* query = const_cast<float*>(query_tensor.ptr<float>());
    float* score = const_cast<float*>(score_tensor.ptr<float>());
    float* output = const_cast<float*>(output_tensor.ptr<float>());
    float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
    float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

    // 线程设置
    constexpr int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE); // 一个block包含256个线程
    dim3 grid(head_num);    // 一个block处理一个头

    // 启动内核
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        // 共享内存用来存储 query 向量，长度为 head_size
        multi_head_attention_kernel<<<grid, block, head_size * sizeof(float), stream_>>>(
            query, score, output, key_cache, value_cache, pos, kv_mul, kv_dim, seq_len,
            head_num, head_size, layer_offset);
    } else {
        multi_head_attention_kernel<<<grid, block, head_size * sizeof(float)>>>(
            query, score, output, key_cache, value_cache, pos, kv_mul, kv_dim, seq_len,
            head_num, head_size, layer_offset);
    }
}

} // namespace kernel