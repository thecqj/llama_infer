#pragma once

#include "op/layer.h"

namespace op {

class MultiHeadAttentionLayer : public Layer {
public:
    explicit MultiHeadAttentionLayer(base::DeviceType device_type, int32_t layer_index,
                                     int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                     int32_t head_num, int32_t head_size);

    base::Status check() const override;

    base::Status forward() override;

private:
    int32_t layer_index_ = 0;   // 当前注意力层在模型中的层级位置
    int32_t pos_ = 0;           // 当前处理的序列位置（自回归逐步递增）
    int32_t kv_mul_ = 0;        // Key/Value 向量的维度扩展因子（GHA）
    int32_t kv_dim_ = 0;        // Key/Value 向量的特征维度
    int32_t seq_len_ = 0;       // 输入序列的长度
    int32_t head_num_ = 0;      // Query 头的数量（标准注意力头数）
    int32_t head_size_ = 0;     // 每个注意力头的内部维度

public:
    void set_layer_index(int32_t layer_idx) {
        layer_index_ = layer_idx;
    }

    void set_pos(int32_t pos) {
        pos_ = pos;
    }
};

} // namespace op