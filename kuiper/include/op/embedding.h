#pragma once

#include "op/layer.h"

namespace op {

class EmbeddingLayer : public LayerParam {
public:
    explicit EmbeddingLayer(base::DeviceType device_type, int32_t seq_len,
                            int32_t vocab_size, int32_t dim);

    base::Status check() const override;

    base::Status forward() override;

private:
    // 嵌入矩阵维度：vocab_size_ x dim_
    int32_t seq_len_ = 0;       // 序列长度
    int32_t vocab_size_ = 0;    // 词汇表大小
    int32_t dim_ = 0;           // 嵌入维度
};

} // namespace op