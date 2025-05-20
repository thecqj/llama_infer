#pragma once

#include <string>
#include <vector>
#include <memory>

#include "base/base.h"
#include "tensor/tensor.h"
#include "base/cuda_config.h"

namespace op {

enum class LayerType : uint8_t {
    kLayerUnknown   = 0,
    kLayerLinear    = 1,
    kLayerEncode    = 2,
    kLayerEmbedding = 3,
    kLayerRMSNorm   = 4,
    kLayerMatmul    = 5,
    kLayerRope      = 6,
    kLayerMHA       = 7,
    kLayerSoftmax   = 8,
    kLayerAdd       = 9,
    kLayerSwiGLU    = 10
};

// ----------------------------------------- 算子基类 -----------------------------------------
class BaseLayer {
public:
    explicit BaseLayer(base::DeviceType device_type, LayerType layer_type,
                       base::DataType data_type, std::string layer_name = "")
            : device_type_{device_type}, layer_type_{layer_type}, 
              data_type_{data_type}, layer_name_{layer_name} {}

    // 初始化参数
    virtual base::Status init() = 0;

    // 检查
    virtual base::Status check() const = 0;

    // 计算函数
    virtual base::Status forward() = 0;

    virtual base::Status forward(const tensor::Tensor& input, const tensor::Tensor& output) = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& output) = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& output) = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& output) = 0;
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& input5, const tensor::Tensor& output) = 0;

    // 输入输出有关函数
    virtual size_t input_size() const = 0;
    virtual size_t output_size() const = 0;

    virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;
    virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

    virtual tensor::Tensor& get_input(int32_t idx) = 0;
    virtual tensor::Tensor& get_output(int32_t idx) = 0;

    virtual const tensor::Tensor& get_input(int32_t idx) const = 0;
    virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

    // 设置权重（有参算子类实现）
    virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight) {
        return base::error::FunctionNotImplement();
    }

    virtual base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr,
                                    base::DeviceType device_type = base::DeviceType::kDeviceUnknown) {
        return base::error::FunctionNotImplement();
    }

protected:
    std::string layer_name_;    // 层名
    LayerType layer_type_ = LayerType::kLayerUnknown;                   // 层类型
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;       // 数据类型
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;   // 设备类型

public:
    // 属性相关函数
    base::DataType data_type() const { return data_type_; }
    LayerType layer_type() const { return layer_type_; }

    const std::string& get_layer_name() const { return layer_name_; }
    void set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }

    base::DeviceType device_type() const { return device_type_; }
    void set_device_type(base::DeviceType device_type) { device_type_ = device_type; }
};

// ----------------------------------------- 无参算子类 -----------------------------------------
class Layer : public BaseLayer {
public:
    explicit Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "",
                   base::DataType data_type = base::DataType::kDataTypeFp32) :
            BaseLayer(device_type, layer_type, data_type, layer_name) {}
    
    // 无需初始化，返回 success
    virtual base::Status init() override { return base::error::Success(); }

    // 检查函数（无具体实现，交给各实际的层）
    virtual base::Status check() const override { return base::error::FunctionNotImplement(); }

    // 计算函数（无具体实现，交给各实际的层）
    virtual base::Status forward() override { return base::error::FunctionNotImplement(); }

    base::Status forward(const tensor::Tensor& input, const tensor::Tensor& output) override;
    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& output) override;
    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& output) override;
    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& output) override;
    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& input5, const tensor::Tensor& output) override;

    // 输入输出有关函数
    size_t input_size() const override { return inputs_.size(); }
    size_t output_size() const override { return outputs_.size(); }

    void set_input(int32_t idx, const tensor::Tensor& input) override;
    void set_output(int32_t idx, const tensor::Tensor& output) override;

    tensor::Tensor& get_input(int32_t idx) override;
    tensor::Tensor& get_output(int32_t idx) override;

    const tensor::Tensor& get_input(int32_t idx) const override;
    const tensor::Tensor& get_output(int32_t idx) const override;
    
protected:
    std::vector<tensor::Tensor> inputs_;    // 输入数据
    std::vector<tensor::Tensor> outputs_;   // 输出数据
    std::shared_ptr<kernel::CudaConfig> cuda_config_;   // cuda 配置

public:
    // 检查相关
    base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                              base::DataType data_type) const;

    template <typename... Args>
    base::Status check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type,
                                       base::DataType data_type, Args&&... args) const;

    // 属性相关
    void reset_input_size(size_t size) { inputs_.resize(size); }
    void reset_output_size(size_t size) { outputs_.resize(size); }

    std::shared_ptr<kernel::CudaConfig> cuda_config() const { return cuda_config_; }
    void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
        if (!config) return;
        cuda_config_ = config;
    }

    // 设备相关
    virtual void to_cuda();
};

template <typename... Args>
inline base::Status 
Layer::check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type,
                             base::DataType data_type, Args&&... args) const {
    // 检查类型
    if (tensor.empty()) {
        return base::error::InvalidArgument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type) {
        return base::error::InvalidArgument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type) {
        return base::error::InvalidArgument("The tensor has a wrong data type.");
    }

    std::vector<int32_t> dims{static_cast<int32_t>(args)...}; // 参数展开
    auto dims_size = tensor.dims_size();

    // 检查维度个数
    if (dims_size != dims.size()) {
        return base::error::InvalidArgument("The tensor has a wrong dims_size.");
    }

    // 检查各维度大小
    for (int i = 0; i < dims_size; ++i) {
        auto dim = tensor.get_dim(i);
        if (dim != dims[i]) {
            return base::error::InvalidArgument("The tensor has a wrong dim shape.");
        }
    }

    return base::error::Success();
}

// ----------------------------------------- 带参算子类 -----------------------------------------
class LayerParam : public Layer {
public:
    explicit LayerParam(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "", 
                        base::DataType data_type = base::DataType::kDataTypeFp32) :
            Layer(device_type, layer_type, layer_name, data_type) {}

    // 设置权重
    base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

    base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
                            const void* weight_ptr,
                            base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;

    // 设备相关
    void to_cuda() override;

protected:
    std::vector<tensor::Tensor> weights_;   // 权重数据

public:
    // 权重相关函数
    tensor::Tensor& get_weight(int32_t idx);
    const tensor::Tensor& get_weight(int32_t idx) const;

    size_t weight_size() const { return weights_.size(); }
    void reset_weight_size(size_t size) { weights_.resize(size); }
};

} // namespace op