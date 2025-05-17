#pragma once
#include <cstdint>

namespace base {

// 设备类型
enum class DeviceType : uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU     = 1,
    kDeviceGPU     = 2
};

enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32    = 1,
    kDataTypeInt32   = 2,
    kDataTypeInt8    = 3
};

class NoCopyable {
protected:
    NoCopyable() = default;
    ~NoCopyable() = default;

    // 禁止拷贝，只允许移动
    NoCopyable(const NoCopyable&) = delete;
    NoCopyable& operator=(const NoCopyable&) = delete;
};


}
