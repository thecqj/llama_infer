#include <glog/logging.h>
#include <numeric>

#include "tensor/tensor.h"

namespace tensor {
// ---------------------------------------- 辅助函数 ----------------------------------------
template <typename T, typename Iterator>
static size_t reduce_dimension(Iterator begin, Iterator end, T init) {
    if (begin >= end) return 0;
    size_t ret = std::accumulate(begin, end, init, std::multiplies<>()); // 累乘
    return ret;
}

// ---------------------------------------- 内存分配函数 ----------------------------------------
// 分配内存（或重新分配内存），确保其底层 buffer_ 满足当前张量的数据大小和设备类型需求
bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
    // 检查 allocator 是否为空
    CHECK_NE(allocator, nullptr) << "The allocator parameter in the allocate function is null pointer";

    // 计算所需字节大小，并检查大小是否为0
    size_t byte_size = this->byte_size();
    CHECK_NE(byte_size, 0) << "The byte_size parameter in the allocate function is equal to zero!";

    // 如果张量大小比内存要小，且每要求重新分配，直接返回
    if (buffer_ && byte_size <= buffer_->byte_size() && !need_realloc) {
        return true;
    } // 走到这，要么要求了重新分配（无论张量多大），要么没要求重新分配但buffer_不能满足张量的空间

    // 分配新的空间，修改buffer_的指向（智能指针会自动释放原内存的空间）
    buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
    CHECK_NE(buffer_->ptr(), nullptr) << "The memory allocated is a null pointer!";

    return true;
}

void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                         bool need_alloc, void* ptr, base::DeviceType device_type) {
    if (!alloc && !need_alloc) {    // 不需要分配器，根据原始指针分配内存（不具有内存归属权）
        const size_t byte_size = base::DataTypeSize(data_type) * size_;
        buffer_ = std::make_shared<base::Buffer>(byte_size, nullptr, ptr, true, device_type);
    } else {    // 使用分配器分配内存
        allocate(alloc, true);  // 由于是初始化，原来没有内存空间，必须强制分配内存
    }
}

// ---------------------------------------- 构造函数 ----------------------------------------
Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr, base::DeviceType device_type)
        : data_type_{data_type} {
    // 设置维度、大小
    dims_.push_back(dim0);
    size_ = dim0;

    // 分配内存创建张量
    // 规定need_alloc与alloc是一致的（要么同时为true，要么同时为false）
    if (need_alloc && alloc) { // 使用分配器分配内存
        allocate(alloc);
    } else {    // 不适用分配器分配内存，根据原始指针分配内存
        if (ptr != nullptr) {
            CHECK(need_alloc == false) // 不具有内存所属权，因此不能强制申请内存
                << "The need_alloc is true when ptr parameter is not a null pointer.";
            init_buffer(nullptr, data_type_, false, ptr, device_type);
        }
        // 走到这，说明创建一个空张量
    }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc,
               void* ptr, base::DeviceType device_type)
        : data_type_{data_type} {
    // 设置维度、大小
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = dim0 * dim1;

    // 分配内存创建张量
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        // 创建二维张量，一定不是空张量
        CHECK(ptr != nullptr) << "The ptr is nullptr when need_alloc is false.";
        CHECK(need_alloc == false) << "The need_alloc is true when ptr parameter is not a null pointer.";
        init_buffer(nullptr, data_type_, false, ptr, device_type);
    }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc,
               void* ptr, base::DeviceType device_type)
        : data_type_{data_type} {
    // 设置维度、大小
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    size_ = dim0 * dim1 * dim2;

    // 分配内存创建张量
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        CHECK(ptr != nullptr) << "The ptr is nullptr when need_alloc is false.";
        CHECK(need_alloc == false) << "The need_alloc is true when ptr parameter is not a null pointer.";
        init_buffer(nullptr, data_type_, false, ptr, device_type);
    }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void* ptr, base::DeviceType device_type)
        : data_type_{data_type} {
    // 设置维度、大小
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    dims_.push_back(dim3);
    size_ = dim0 * dim1 * dim2 * dim3;

    // 分配内存创建张量
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        CHECK(ptr != nullptr) << "The ptr is nullptr when need_alloc is false.";
        CHECK(need_alloc == false) << "The need_alloc is true when ptr parameter is not a null pointer.";
        init_buffer(nullptr, data_type_, false, ptr, device_type);
    }
}

Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr, base::DeviceType device_type)
        : data_type_{data_type}, dims_{dims} {
    // 求大小
    size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
        
    // 分配内存创建张量
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        CHECK(ptr != nullptr) << "The ptr is nullptr when need_alloc is false.";
        CHECK(need_alloc == false) << "The need_alloc is true when ptr parameter is not a null pointer.";
        init_buffer(nullptr, data_type_, false, ptr, device_type);
    }
}

// ---------------------------------------- 内存有关函数 ----------------------------------------


// ---------------------------------------- 张量有关函数 ----------------------------------------
std::vector<int32_t> Tensor::strides() const {
    if (dims_.empty()) return {};
    std::vector<int32_t> strides;
    // tensor: [d0, d1, d2] -> strides: [d1 * d2, d2, 1]
    for (size_t i = 0; i < dims_.size() - 1; ++i) {
        size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
        strides.push_back(stride);
    }
    strides.push_back(1);
    return strides;
}

void Tensor::reshape(const std::vector<int32_t>& dims) {
    size_t new_size = reduce_dimension(dims.begin(), dims.end(), 1);

    // 如果是空张量，只更新维度信息，不实际分配内存（交给用户）
    if (!buffer_) {
        dims_ = dims;
        size_ = new_size;
        return;
    }

    if (new_size > size_) { // 需要扩容
        auto byte_size = new_size * base::DataTypeSize(data_type_);
        // 分配新的内存
        auto new_buffer = std::make_shared<base::Buffer>(byte_size, buffer_->allocator());
        CHECK(new_buffer->allocate());
        new_buffer->copy_from(buffer_.get()); // 拷贝原数据
        buffer_ = new_buffer;
    }
    // 不需要扩容就还是使用原来的内存

    // 更新维度信息
    dims_ = dims;
    size_ = new_size;
}

Tensor Tensor::clone() const {  // 深拷贝
    // 空张量不需要分配内存，直接返回
    if (!buffer_) return *this;

    // 计算内存大小，获取分配器
    auto byte_size = this->byte_size();
    auto allocator = buffer_->allocator();

    // 拷贝构造创建一个对象（浅拷贝）
    Tensor new_tensor = *this;

    // 如果原张量不具有内存（externel == true, allocator == nullptr），则取一个分配器
    if (!allocator) {
        if (buffer_->device_type() == base::DeviceType::kDeviceCPU) {
            allocator = base::CPUDeviceAllocatorFactory::get_instance();
        } else if (buffer_->device_type() == base::DeviceType::kDeviceGPU) {
            allocator = base::CUDADeviceAllocatorFactory::get_instance();
        } else {
            // 记录错误日志
            LOG(ERROR) << "Unknown DeviceType.";
        }
    }

    // 分配新的内存
    new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
    new_tensor.buffer_->copy_from(buffer_.get());   // 拷贝原数据

    return new_tensor;
}

} // namespace tensor