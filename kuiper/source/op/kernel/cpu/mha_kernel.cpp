#include <glog/logging.h>
#include <armadillo>

#include "mha_kernel.h"
#include "matmul_kernel.h"
#include "softmax_kernel.h"
#include "scale_sum_kernel.h"

namespace kernel {

void mha_kernel(const int32_t layer_index, const int32_t pos, const int32_t kv_mul, const int32_t kv_dim,
                const int32_t seq_len, const int32_t head_num, const int32_t head_size,
                const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                const tensor::Tensor& output_tensor, void* stream) {
    // 计算当前层在缓存中的偏移量
    int32_t layer_offset = layer_index * seq_len * kv_dim;
    float scale = 1.f / std::sqrt(static_cast<float>(head_size));

    // 逐头处理
    for (int32_t h = 0; h < head_num; ++h) {
        // 取该头下的score（输入是第pos个token的结果）
        float* score_head_addr = const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);

        // 取该头下的query（输入是第pos个token）
        float* query_head_addr = const_cast<float*>(query_tensor.ptr<float>() + h * head_size);
        tensor::Tensor query_mat(base::DataType::kDataTypeFp32, head_size, false, nullptr, query_head_addr);
        query_mat.set_device_type(base::DeviceType::kDeviceCPU);

        // 使用 KV-Cache 逐个处理K的每个 token
        for (int32_t t = 0; t <= pos; ++t) {
            // 定位Key缓存位置
            int32_t cache_offset = t * kv_dim + (h / kv_mul) * head_size;

            // 取该头下的key的第t行
            float* key_head_addr = const_cast<float*>(key_cache_tensor.ptr<float>() + layer_offset + cache_offset);
            tensor::Tensor key_mat(base::DataType::kDataTypeFp32, 1, head_size, false, nullptr, key_head_addr);
            key_mat.set_device_type(base::DeviceType::kDeviceCPU);

            // 取该头下的score的第t列
            tensor::Tensor score_mat(base::DataType::kDataTypeFp32, 1, false, nullptr, score_head_addr + t);
            score_mat.set_device_type(base::DeviceType::kDeviceCPU);

            // 做向量点积，存入到score_mat中，会更新score_head_addr
            matmul_kernel_cpu(query_mat, key_mat, score_mat, scale, stream);
        }

        // 取得到的一行score
        tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, 1, pos + 1, false, nullptr, score_head_addr);
        score_head_tensor.set_device_type(base::DeviceType::kDeviceCPU);

        // 做 softmax
        softmax_inplace_cpu(score_head_tensor);

        // 定位value缓存位置
        int32_t cache_offset = (h / kv_mul) * head_size;
        float* value_head_addr = const_cast<float*>(value_cache_tensor.ptr<float>()) + layer_offset + cache_offset;
        // 转为tensor：(pos + 1) x head_size 的矩阵
        tensor::Tensor value_head_tensor(base::DataType::kDataTypeFp32, pos + 1, head_size, false, nullptr, value_head_addr);
        value_head_tensor.set_device_type(base::DeviceType::kDeviceCPU);

        // 定位 output（输入是该第pos个token与V运算的结果）
        float* output_head_addr = const_cast<float*>(output_tensor.ptr<float>()) + h * head_size;
        // 初始化置零（因为score矩阵是一个下三角，其余部分都为0）
        auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
        allocator->memset_zero(output_head_addr, sizeof(float) * head_size, nullptr, false);
        // 转为tensor
        tensor::Tensor output_head_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr, output_head_addr);
        output_head_tensor.set_device_type(base::DeviceType::kDeviceCPU);

        // 做矩阵乘法（每列的加权求和）
        scale_sum_kernel_cpu(value_head_tensor, score_head_tensor, output_head_tensor, pos, head_size, kv_dim);
    }
}

} // namespace kernel