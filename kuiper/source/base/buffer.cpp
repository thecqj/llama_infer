#include <glog/logging.h>

#include "base/buffer.h"

namespace base {

Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, 
               void* ptr, bool use_externel)
        : byte_size_{byte_size}, allocator_{allocator}, 
          ptr_{ptr}, use_externel_{use_externel} {
    if (!ptr_ && allocator_) {
        // 如果只是传了分配器过来，需要用它来初始化其他信息
        device_type_ = allocator_->device_type();
        use_externel_ = false;
        // 使用分配器申请内存
        ptr_ = allocator_->allocate(byte_size_);
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

    use_externel_ = false;
    ptr_ = allocator_->allocate(byte_size_);

    if (!ptr_) return false;
    return true;
}

void Buffer::copy_from(const Buffer& buffer) {
    CHECK_NE(buffer.ptr_, nullptr);
    CHECK_NE(allocator_, nullptr);

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