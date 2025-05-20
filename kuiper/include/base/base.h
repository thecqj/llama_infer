#pragma once

#include <glog/logging.h>
#include <cstdint>
#include <string>

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

// 状态码
enum StatusCode: int8_t {
    kSuccess             = 0,
    kFunctionUnImplement = 1,
    kPathNotValid        = 2,
    kModelParseError     = 3,
    kInternalError       = 4,
    kKeyValueHasExist    = 5,
    kInvalidArgument     = 6
};

class Status {
public:
    Status(int code = StatusCode::kSuccess, std::string err_message = "") 
        : code_{code}, message_{err_message} {}

    Status(const Status& other) = default;
    Status& operator=(const Status& other) = default;

    Status& operator=(int code) {
        code_ = code;
        return *this;
    }

    bool operator==(int code) const { return code_ == code; }
    bool operator!=(int code) const { return !this->operator==(code); }

    operator int() const { return code_; }
    operator bool() const { return code_ == StatusCode::kSuccess; }

    int32_t get_err_code() const { return code_; }
    const std::string& get_err_msg() const { return message_; }
    void set_err_msg(const std::string& err_msg) { message_ = err_msg; }

private:
    int code_ = StatusCode::kSuccess;
    std::string message_;
};

namespace error {

#define STATUS_CHECK(call)                                                                       \
    do {                                                                                         \
        const base::Status& status = call;                                                       \
        if (!status) {                                                                           \
            const size_t buf_size = 512;                                                         \
            char buf[buf_size];                                                                  \
            snprintf(buf, buf_size - 1,                                                          \
                     "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
                     __LINE__, int(status), status.get_err_msg().c_str());                       \
            LOG(FATAL) << buf;                                                                   \
        }                                                                                        \
    } while(0)

Status Success(const std::string& err_msg = "");

Status FunctionNotImplement(const std::string& err_msg = "");

Status PathNotValid(const std::string& err_msg = "");

Status ModelParseError(const std::string& err_msg = "");

Status InternalError(const std::string& err_msg = "");

Status KeyHasExits(const std::string& err_msg = "");

Status InvalidArgument(const std::string& err_msg = "");

} // namespace error

} // namespace base