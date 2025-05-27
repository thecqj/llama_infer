#pragma once

#include "op/layer.h"

namespace op {

// 矩阵向量乘 gemv：y = x · A^T (== A · x^T) + b
// x: 1    x dim1
// A: dim0 x dim1
// b: 1    x dim0
// y: 1    x dim0
class MatmulLayer : public LayerParam {
public:
    explicit MatmulLayer(base::DeviceType device_type, base::DataType data_type, 
                         int32_t dim0, int32_t dim1, bool is_quant_layer = false,
                         bool has_bias = false);

    base::Status check() const override;

    base::Status forward() override;

private:
    int32_t dim0_ = 0;  // 权重另一维度（== 偏置维度）
    int32_t dim1_ = 0;  // 向量维度

    bool has_bias_ = false;
    std::vector<tensor::Tensor> bias_;  // 偏置张量

public:
    base::Status set_bias(int32_t idx, const tensor::Tensor& bias);
    base::Status set_bias(int32_t idx, int32_t& dim, const void* bias_ptr, base::DeviceType device_type);

    tensor::Tensor& get_bias(int32_t idx);
    const tensor::Tensor& get_bias(int32_t idx) const;

    void to_cuda() override;
};

} // namespace op