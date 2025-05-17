#pragma once
#include <cstdint>

namespace base {

// 设备类型
enum class DeviceType : uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU     = 1,
    kDeviceGPU     = 2
};

// 数据类型
enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32    = 1,
    kDataTypeInt32   = 2,
    kDataTypeInt8    = 3
};

// 数据类型的大小
inline size_t DataTypeSize(DataType data_type) {
    switch (data_type) {
        case DataType::kDataTypeFp32:
            return sizeof(float);
        case DataType::kDataTypeInt32:
            return sizeof(int32_t);
        case DataType::kDataTypeInt8:
            return sizeof(int8_t);
        default:
            return 0;
    }
}

class NoCopyable {
protected:
    NoCopyable() = default;
    ~NoCopyable() = default;

    // 禁止拷贝，只允许移动
    NoCopyable(const NoCopyable&) = delete;
    NoCopyable& operator=(const NoCopyable&) = delete;
};


}
