#include <cstdio>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "base/alloc.h"

namespace base {

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    if (byte_size == 0) return nullptr;

    void* ptr = nullptr;
    cudaError_t state = cudaMalloc(&ptr, byte_size);

    if (state != cudaSuccess) {
        // 记录日志，分配失败并返回
        char buf[256];
        snprintf(buf, 256,
                "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
                "left on  device.",
                byte_size >> 20);
        LOG(ERROR) << buf;
        return nullptr;
    }
    return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
    if (ptr == nullptr) return;
    
    cudaError_t state = cudaFree(ptr);
    // 如果释放失败，记录日志，并退出程序
    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

} // namespace base