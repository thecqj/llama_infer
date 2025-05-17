#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/alloc.h"

TEST(test_cpu_allocator, allocate1) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    float* ptr = (float*)alloc->allocate(16 * sizeof(float)); // 64B
    CHECK_NE(ptr, nullptr);
    alloc->release(ptr);
}

TEST(test_cpu_allocator, allocate2) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    float* ptr = (float*)alloc->allocate(1600 * sizeof(float)); // 6400B
    CHECK_NE(ptr, nullptr);
    alloc->release(ptr);
}

TEST(test_cpu_allocator, memset_zero) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    const size_t byte_size = 2 * sizeof(float);
    float* ptr = (float*)alloc->allocate(byte_size);
    ptr[0] = 1; ptr[1] = 2;

    alloc->memset_zero(ptr, byte_size);
    CHECK_EQ(ptr[0], 0);
    CHECK_EQ(ptr[1], 0);
    alloc->release(ptr);
}

TEST(test_cpu_allocator, memcpy) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    const size_t byte_size = 2 * sizeof(float);
    float* ptr = (float*)alloc->allocate(byte_size);
    ptr[0] = 1; ptr[1] = 2;
    float* t = (float*)alloc->allocate(byte_size);

    alloc->memcpy(t, ptr, byte_size);
    CHECK_NE(t, ptr);
    CHECK_EQ(t[0], ptr[0]);
    CHECK_EQ(t[1], ptr[1]);
    alloc->release(t);
    alloc->release(ptr);
}