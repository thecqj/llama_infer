#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "base/alloc.h"

TEST(test_gpu_allocator, allocate) {
    using namespace base;
    auto alloc = CUDADeviceAllocatorFactory::get_instance();
    const size_t byte_size = 2 * sizeof(float);

    float* d_p = (float*)alloc->allocate(byte_size);
    alloc->memset_zero(d_p, byte_size);

    float* h_p = (float*)malloc(byte_size);
    alloc->memcpy(h_p, d_p, byte_size, MemcpyKind::kMemcpyCUDA2CPU);
    
    CHECK_EQ(h_p[0], 0);
    CHECK_EQ(h_p[1], 0);

    alloc->release(d_p);
    free(h_p);
}