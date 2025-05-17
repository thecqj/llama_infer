#pragma once

#include <cstddef>
#include <vector>
#include <cuda_runtime.h>

#include "base/buffer.h"

namespace tensor {

class Tensor {
public:
    explicit Tensor() = default;

    // 一维张量构造
    explicit Tensor(base::DataType data_type, int32_t dim0,
                    bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    // 二维张量构造
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
                    bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    // 三维张量构造
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                    bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    // 四维张量构造
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                    bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    // 多维张量构造
    explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, 
                    bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

private:
    size_t size_ = 0;                                 // 张量中数据个数
    std::vector<int32_t> dims_;                       // 张量各维度大小
    std::shared_ptr<base::Buffer> buffer_ = nullptr;  // 存储内存
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;   // 张量的数据类型

private:
    // 初始化内存
    void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                     bool need_alloc, void* ptr);

public:
    // 分配内存，或重新分配内存
    bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

    // // 内存数据迁移
    // void to_cpu();
    // void to_cuda(cudaStream_t stream = nullptr);

    // // 重新设置内存
    // bool assign(std::shared_ptr<base::Buffer> buffer);

    // // 重置张量的数据类型和维度，可能触发内存重新分配
    // void reset(base::DataType data_type, const std::vector<int32_t>& dims);

    // // 设置张量数据所在的设备类型，可能需要迁移数据
    // void set_device_type(base::DeviceType device_type);

    // // 返回当前数据所在的设备类型
    // base::DeviceType device_type() const;

public:
    // // 返回指向张量数据的指针
    // template <typename T> T* ptr();
    // template <typename T> const T* ptr() const;

    // // 返回指定线性索引位置的数据指针
    // template <typename T> T* ptr(int64_t index);
    // template <typename T> const T* ptr(int64_t index) const;
    // template <typename T> T& index(int64_t offset);
    // template <typename T> const T& index(int64_t offset) const;

    // // 调整张量维度
    // void reshape(const std::vector<uint32_t>& dims);

    // 张量属性查询
    bool empty() const { return !size_ || !buffer_ || !buffer_->ptr(); }    // 检查张量是否为空
    size_t size() const { return size_; }   // 张量元素个数
    size_t byte_size() const { return size() * DataTypeSize(data_type_); }   // 张量所占用的字节数
    int32_t dims_size() const { return static_cast<int32_t>(dims_.size()); } // 张量的维度
    base::DataType data_type() const { return data_type_; } // 张量的数据类型
    int32_t get_dim(int32_t idx) const {    // 张量某个维度的大小
        CHECK_GE(idx, 0);
        CHECK_LT(idx, dims_.size());
        return static_cast<int32_t>(dims_[idx]);
    }
    const std::vector<int32_t>& dims() const { return dims_; }  // 张量维度的常量引用
    // std::vector<size_t> strides() const;        // 计算并返回各维度的步长（字节数）
    std::shared_ptr<base::Buffer> get_buffer() const { return buffer_; }    // 返回指向内存的指针

    // // 创建当前张量的深拷贝
    // tensor::Tensor clone() const;
};

} // namespace tensor