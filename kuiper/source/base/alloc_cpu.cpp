#include <cstdlib>

#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KUIPER_HAVE_POSIX_MEMALIGN
#endif

namespace base {

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
    if (byte_size == 0) return nullptr;

#ifdef KUIPER_HAVE_POSIX_MEMALIGN
    void* data = nullptr;
    // 对较大内存块（≥1KB）使用更高的对齐（32 字节），适配 SIMD 或缓存行优化
    const size_t alignment = byte_size >= size_t(1024) ? size_t(32) : size_t(16);
    int status = posix_memalign((void**)&data, 
                                alignment >= sizeof(void*) ? alignment : sizeof(void*),
                                byte_size);
    if (status != 0) return nullptr;    // 分配成功返回0，否则返回非0
    return data;
#else
    void* data = malloc(byte_size);
    return data;
#endif
}

void CPUDeviceAllocator::release(void* ptr) const {
    if (ptr == nullptr) return;
    free(ptr);
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;

} // namespace base