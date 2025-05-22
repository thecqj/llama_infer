#pragma once

#include "op/layer.h"

namespace op {

class RMSNormLayer : public LayerParam {
public:
    explicit RMSNormLayer(base::DeviceType device_type, base::DataType data_type);

    base::Status check() const override;

    base::Status forward() override;
};

} // namespace op