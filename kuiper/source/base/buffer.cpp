#include <glog/logging.h>

#include "base/buffer.h"

namespace base {

Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, 
               void* ptr, bool use_externel, DeviceType device_type)
        : byte_size_{byte_size}, allocator_{allocator}, 
          ptr_{ptr}, use_externel_{use_externel}, device_type_{device_type} {

    if (use_externel == true) {
        // 检查不具有归属权时需要满足的条件，ptr不能为空。若allocator不空可以在将来使用
        CHECK(ptr != nullptr) <<
            "ptr is nullptr when use_externel is true";
    } else {
        // 检查具有归属权时需要满足的条件，allocator不能为空
        CHECK(allocator != nullptr) << "allocator is nullptr when use_externel is false";

        // 无论ptr是否为空，都应申请内存
        ptr_ = allocator_->allocate(byte_size_);
        device_type_ = allocator_->device_type();
    }
}

Buffer::~Buffer() {
    if (!use_externel_) {   // 拥有内存的归属权才释放
        if (ptr_ && allocator_) {
            allocator_->release(ptr_);
            ptr_ = nullptr;
        }
    }
}

bool Buffer::allocate() {
    if (!allocator_ || byte_size_ == 0) return false;

    // 根据是否有归属权决定要不要释放原内存
    if (use_externel_ == true) {
        use_externel_ = false;  // 更新
    } else {
        allocator_->release(ptr_);  // 释放内存（可能导致悬空指针）
    }

    // 分配新的内存
    ptr_ = allocator_->allocate(byte_size_);

    if (!ptr_) return false;
    return true;
}

void Buffer::copy_from(const Buffer& buffer) {
    CHECK_NE(buffer.ptr_, nullptr);
    CHECK_NE(allocator_, nullptr);  // 无论是否有归属权，都会拷贝内存

    const size_t byte_size = std::min(byte_size_, buffer.byte_size_);
    const DeviceType& buffer_device = buffer.device_type();
    const DeviceType& current_device = device_type();
    CHECK(buffer_device != DeviceType::kDeviceUnknown &&
          current_device != DeviceType::kDeviceUnknown);

    MemcpyKind memcpyKind;
    if (buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCPU) {
        memcpyKind = MemcpyKind::kMemcpyCPU2CPU;
    } else if (buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceGPU) {
        memcpyKind = MemcpyKind::kMemcpyCPU2CUDA;
    } else if (buffer_device == DeviceType::kDeviceGPU && current_device == DeviceType::kDeviceCPU) {
        memcpyKind = MemcpyKind::kMemcpyCUDA2CPU;
    } else if (buffer_device == DeviceType::kDeviceGPU && current_device == DeviceType::kDeviceGPU) {
        memcpyKind = MemcpyKind::kMemcpyCUDA2CUDA;
    } else {
        // 记录日志，并退出程序
        LOG(ERROR) << "Unknown MemcpyKind!";
    }

    allocator_->memcpy(ptr_, buffer.ptr_, byte_size, memcpyKind);
}

void Buffer::copy_from(const Buffer* buffer) {
    CHECK_NE(buffer, nullptr);
    CHECK_NE(buffer->ptr_, nullptr);
    CHECK_NE(allocator_, nullptr);

    const size_t byte_size = std::min(byte_size_, buffer->byte_size_);
    const DeviceType& buffer_device = buffer->device_type();
    const DeviceType& current_device = device_type();
    CHECK(buffer_device != DeviceType::kDeviceUnknown &&
          current_device != DeviceType::kDeviceUnknown);

    MemcpyKind memcpyKind;
    if (buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCPU) {
        memcpyKind = MemcpyKind::kMemcpyCPU2CPU;
    } else if (buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceGPU) {
        memcpyKind = MemcpyKind::kMemcpyCPU2CUDA;
    } else if (buffer_device == DeviceType::kDeviceGPU && current_device == DeviceType::kDeviceCPU) {
        memcpyKind = MemcpyKind::kMemcpyCUDA2CPU;
    } else if (buffer_device == DeviceType::kDeviceGPU && current_device == DeviceType::kDeviceGPU) {
        memcpyKind = MemcpyKind::kMemcpyCUDA2CUDA;
    } else {
        // 记录日志，并退出程序
        LOG(ERROR) << "Unknown MemcpyKind!";
        return;
    }

    allocator_->memcpy(ptr_, buffer->ptr_, byte_size, memcpyKind);
}

}