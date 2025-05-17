#include <glog/logging.h>

#include "tensor/tensor.h"

namespace tensor {

// ---------------------------------------- 内存分配函数 ----------------------------------------
// 分配内存（或重新分配内存），确保其底层 buffer_ 满足当前张量的数据大小和设备类型需求
bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
    // 检查 allocator 是否为空
    if (!allocator) {
        LOG(ERROR) << "The allocator parameter in the allocate function is null pointer";
        return false;
    }

    // 计算所需字节大小，并检查大小是否为0
    size_t byte_size = this->byte_size();
    if (byte_size == 0) {
        LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
        return false;
    }

    // 判断是否需要重新分配
    if (buffer_ && byte_size <= buffer_->byte_size()) {
        
    }

}

void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                         bool need_alloc, void* ptr) {
    if (!alloc && !need_alloc) {

    }
}

// ---------------------------------------- 构造函数 ----------------------------------------
Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
        : data_type_{data_type} {
    // 设置维度、大小
    dims_.push_back(dim0);
    size_ = dim0;

    // 分配内存创建张量
    if (need_alloc && alloc) { // 使用分配器分配内存
        allocate(alloc);
    } else {    // 不适用分配器分配内存
        if (ptr) {
            CHECK(need_alloc == false) 
                << "The need_alloc is true when ptr parameter is not a null pointer.";
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }
}

} // namespace tensor