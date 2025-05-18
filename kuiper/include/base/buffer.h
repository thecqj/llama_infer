#pragma once

#include <memory>

#include "base/alloc.h"

namespace base {

class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
public:
    explicit Buffer() = default;
    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                    void* ptr = nullptr, bool use_externel = false,
                    DeviceType device_type = DeviceType::kDeviceUnknown);
    virtual ~Buffer();
    
    bool allocate();
    void copy_from(const Buffer&);
    void copy_from(const Buffer*);

private:
    size_t byte_size_ = 0;                                  // 内存大小
    void* ptr_ = nullptr;                                   // 内存指针
    std::shared_ptr<DeviceAllocator> allocator_ = nullptr;  // 分配器
    bool use_externel_ = false;                             // 是否对该块内存有归属权
    DeviceType device_type_ = DeviceType::kDeviceUnknown;   // 设备类型

public:
    size_t byte_size() const { return byte_size_; }

    void* ptr() { return ptr_; }
    const void* ptr() const { return ptr_; }

    std::shared_ptr<DeviceAllocator> allocator() const { return allocator_; }

    bool is_externel() const { return use_externel_; }

    DeviceType device_type() const { return device_type_; }
    void set_device_type(DeviceType device_type) { device_type_ = device_type; }

    std::shared_ptr<Buffer> get_shared_from_this() { return shared_from_this(); }
};

} // namespace base