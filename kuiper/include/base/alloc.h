#pragma once

#include <cstddef>
#include <memory>

#include "base/base.h"

namespace base {

// 内存移动方向
enum class MemcpyKind {
    kMemcpyCPU2CPU   = 0,
    kMemcpyCPU2CUDA  = 1,
    kMemcpyCUDA2CPU  = 2,
    kMemcpyCUDA2CUDA = 3
};

// 内存分配器（抽象基类）
class DeviceAllocator {
public:
    // 构造函数指明所属设备类型
    explicit DeviceAllocator(DeviceType device_type) : device_type_{device_type} {}

    virtual DeviceType device_type() const { return device_type_; }

    // 申请、释放
    virtual void* allocate(size_t byte_size) const = 0;
    virtual void release(void* ptr) const = 0;

    virtual void memcpy(void* dest_ptr, const void* src_ptr, size_t byte_size,
                        // 后三个参数供CUDA使用
                        MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,
                        void* stream = nullptr, 
                        bool need_sync = false) const;
    
    virtual void memset_zero(void* ptr, size_t byte_size, 
                             void* stream = nullptr, bool need_sync = false);

private:
    DeviceType device_type_ = DeviceType::kDeviceUnknown;    // 默认值
};

// CPU内存分配器
class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
};

// CUDA内存分配器
class CUDADeviceAllocator: public DeviceAllocator {
public:
    explicit CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceGPU) {}

    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
};

// 工厂模式
class CPUDeviceAllocatorFactory {
public:
    static std::shared_ptr<CPUDeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CPUDeviceAllocator>();
        }
        return instance;
    }

private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CUDADeviceAllocatorFactory {
public:
    static std::shared_ptr<CUDADeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CUDADeviceAllocator>();
        }
        return instance;
    }

private:
    static std::shared_ptr<CUDADeviceAllocator> instance;
};

} // namespace base
