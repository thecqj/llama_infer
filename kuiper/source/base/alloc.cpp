#include <cstring>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "base/alloc.h"

namespace base {

void DeviceAllocator::memcpy(void* dest_ptr, const void* src_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
    CHECK_NE(src_ptr, nullptr);
    CHECK_NE(dest_ptr, nullptr);

    if (byte_size == 0) return;

    cudaStream_t stream_ = nullptr;
    if (stream) {
        // stream 不空（即非默认流），将其转换成 cudaStream_t（CUstream_st* 的别名），并初始化 stream_
        stream_ = static_cast<cudaStream_t>(stream);
    }

    // CPU 之间传输直接使用 memcpy
    // 若使用默认流，使用 cudaMemcpy；否则，使用 cudaMemcpyAsync
    if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
        std::memcpy(dest_ptr, src_ptr, byte_size);
    } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
        if (!stream_) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
        }
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
        if (!stream_) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
        }
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
        if (!stream_) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
        }
    } else {
        // 记录日志，并直接退出
        LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
    }

    // 需要线程同步则进行同步
    if (need_sync) {
        cudaDeviceSynchronize();
    }
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync) {
    CHECK(device_type_ != base::DeviceType::kDeviceUnknown);

    // 若是cpu，则直接使用 memset
    // 若是默认流，使用cudaMemset；否则，使用cudaMemsetAsync
    if (device_type_ == DeviceType::kDeviceCPU) {
        std::memset(ptr, 0, byte_size);
    } else {
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            cudaMemsetAsync(ptr, 0, byte_size, stream_);
        } else {
            cudaMemset(ptr, 0, byte_size);
        }

        if (need_sync) {
            cudaDeviceSynchronize();
        }
    }
}

} // namespace base