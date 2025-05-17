#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/buffer.h"

TEST(test_buffer, alloc_1) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    Buffer buffer(32, alloc);   // use_externel == false, allocator != nullptr, ptr == nullptr

    ASSERT_EQ(buffer.is_externel(), false);
    ASSERT_NE(buffer.ptr(), nullptr);
}

TEST(test_buffer, alloc_2) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    float* ptr = new float{3.14};
    Buffer buffer(4, alloc, ptr);   // use_externel == false, allocator != nullptr, ptr != nullptr

    ASSERT_EQ(buffer.is_externel(), false);
    ASSERT_NE(buffer.ptr(), ptr);
    
    delete ptr;
}

TEST(test_buffer, use_externel) {
    using namespace base;

    float* ptr = new float[32];
    *ptr = 5;
    Buffer buffer(32, nullptr, ptr, true);  // use_externel == true, ptr != nullptr, allocator == nullptr
    auto buffer_ptr = static_cast<float*>(buffer.ptr());

    ASSERT_EQ(buffer.is_externel(), true);
    ASSERT_EQ(*buffer_ptr, *ptr);

    delete[] ptr;
}

TEST(test_buffer, allocate_1) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    Buffer buffer(4, alloc);
    *static_cast<float*>(buffer.ptr()) = 3.14;

    buffer.allocate();
    auto ptr = static_cast<float*>(buffer.ptr());
    *ptr = 5;

    ASSERT_EQ(buffer.is_externel(), false);
    ASSERT_EQ(*static_cast<float*>(buffer.ptr()), 5);
}

TEST(test_buffer, allocate_2) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    float* ptr = new float{3.14};
    Buffer buffer(4, alloc, ptr, true); // use_externel == true, ptr != nullptr, allocator != nullptr
    ASSERT_EQ(buffer.is_externel(), true);

    buffer.allocate();
    *static_cast<float*>(buffer.ptr()) = 5;

    ASSERT_EQ(buffer.is_externel(), false);
    ASSERT_EQ(*static_cast<float*>(buffer.ptr()), 5);
}

TEST(test_buffer, copy_from_1) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();

    Buffer buffer_from(4, alloc);
    auto ptr_from = static_cast<float*>(buffer_from.ptr());
    *ptr_from = 5;

    Buffer buffer_to(4, alloc);
    buffer_to.copy_from(buffer_from);   // cpu -> cpu
    ASSERT_NE(buffer_to.ptr(), buffer_from.ptr());
    
    auto ptr_to = static_cast<float*>(buffer_to.ptr());
    ASSERT_EQ(*ptr_to, *ptr_from);
}

TEST(test_buffer, copy_from_2) {
    using namespace base;
    auto cpu_alloc = CPUDeviceAllocatorFactory::get_instance();
    auto gpu_alloc = CUDADeviceAllocatorFactory::get_instance();

    Buffer buffer_from(4, cpu_alloc);
    auto ptr_from = static_cast<float*>(buffer_from.ptr());
    *ptr_from = 5;

    Buffer buffer_to(4, gpu_alloc);
    buffer_to.copy_from(buffer_from);   // cpu -> gpu

    Buffer buffer_tmp(4, gpu_alloc);
    buffer_tmp.copy_from(buffer_to);    // gpu -> gpu

    Buffer h_buffer(4, cpu_alloc);
    h_buffer.copy_from(buffer_tmp);  // gpu -> cpu

    auto ptr = static_cast<float*>(h_buffer.ptr());
    ASSERT_EQ(*ptr, *ptr_from);
}