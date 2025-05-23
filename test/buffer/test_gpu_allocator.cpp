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

TEST(test_gpu_allocator, multi_alloc) {
    using namespace base;
    auto alloc = CUDADeviceAllocatorFactory::get_instance();

    srand(time(nullptr));

    for (int epoch = 0; epoch < 10; ++epoch) {
        std::vector<void*> ptrs;
        for (int i = 0; i < 100; ++i) {
            int byte_size = rand() % (2 * 1024 * 1024) + 1;
            ptrs.push_back(alloc->allocate(byte_size));
        }
        for (int i = 0; i < 100; ++i) {
            alloc->release(ptrs[i]);
        }
    }

}