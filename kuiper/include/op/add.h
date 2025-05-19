#pragma once

#include "op/layer.h"

namespace op {

// 继承无参算子类，定义检查、计算函数
class VecAddLayer : public Layer {
public:
    explicit VecAddLayer(base::DeviceType device_type);

    base::Status check() const override;

    base::Status forward() override;
};

} // namespace op