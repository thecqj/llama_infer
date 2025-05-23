#include <cstdio>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "base/alloc.h"

namespace base {

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    if (byte_size == 0) return nullptr;

    int id = -1;
    cudaError_t state = cudaGetDevice(&id); // 获取设备编号

    // 大块内存的分配
    if (byte_size > 1024 * 1024) {
        auto& big_buffers = big_buffers_map_[id];
        // 找最合适的一块
        int select_id = -1;
        for (int i = 0; i < big_buffers.size(); ++i) {
            // 满足三个条件：比byte_size大、没被占用、剩余的空间需小于1MB
            if (big_buffers[i].byte_size_ >= byte_size && !big_buffers[i].busy_ &&
                big_buffers[i].byte_size_ - byte_size < 1 * 1024 * 1024) {
                // 选择满足条件下最小的一个
                if (select_id == -1 || big_buffers[i].byte_size_ < big_buffers[select_id].byte_size_) {
                    select_id = i;
                }
            }
        }
        // 找到了，返回指针
        if (select_id != -1) {
            big_buffers[select_id].busy_ = true;
            return big_buffers[select_id].data_;
        }
        // 没找到，调用cudaMalloc，并对新分配的内存块进行管理
        void* ptr = nullptr;
        state = cudaMalloc(&ptr, byte_size);
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
        big_buffers.emplace_back(ptr, byte_size, true); // 对新的内存块做管理
        return ptr;
    }

    // 小块内存的分配
    auto& cuda_buffers = cuda_buffers_map_[id];
    for (int i = 0; i < cuda_buffers.size(); ++i) {
        // 满足两个条件：比byte_size大、没被占用
        if (cuda_buffers[i].byte_size_ >= byte_size && !cuda_buffers[i].busy_) {
            cuda_buffers[i].busy_ = true;
            no_busy_cnt_[id] -= cuda_buffers[i].byte_size_; // 空闲内存也要减少
            return cuda_buffers[i].data_;   // 找到了直接返回
        }
    }
    // 没找打，使用cudaMalloc
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
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
    cuda_buffers.emplace_back(ptr, byte_size, true);    // 对新的内存块做管理
    return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
    if (ptr == nullptr) return;
    
    cudaError_t state = cudaSuccess;
    
    // 清理内存块，防止堆积过多
    for (auto& it : cuda_buffers_map_) {
        if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {  // 大于阈值1GB就清理一次
            auto& id = it.first;
            auto& cuda_buffers = it.second;
            std::vector<CudaMemoryBuffer> temp; // 存放非空闲的内存块
            for (int i = 0; i < cuda_buffers.size(); ++i) {
                if (!cuda_buffers[i].busy_) {   // 如果是空闲的就释放掉
                    state = cudaFree(cuda_buffers[i].data_);
                    CHECK(state == cudaSuccess) 
                        << "Error: CUDA error when release memory on device " << id;
                } else {
                    temp.push_back(cuda_buffers[i]);    // 非空闲加入到 temp 中
                }
            }
            // 重置内存池
            cuda_buffers.clear();
            cuda_buffers = temp;
            no_busy_cnt_[id] = 0;
        }
    }

    // 遍历每个设备的每个内存块，直到找到了对应的内存块
    for (int id = 0; id < cuda_buffers_map_.size(); ++id) {
        // 找小块内存
        auto& cuda_buffers = cuda_buffers_map_[id];
        for (int i = 0; i < cuda_buffers.size(); ++i) {
            if (cuda_buffers[i].data_ == ptr) {
                cuda_buffers[i].busy_ = false;
                no_busy_cnt_[id] += cuda_buffers[i].byte_size_;   // 空闲内存随之增加
                return;
            }
        }
        // 找大块内存
        auto& big_buffers = big_buffers_map_[id];
        for (int i = 0; i < big_buffers.size(); ++i) {
            if (big_buffers[i].data_ == ptr) {
                big_buffers[i].busy_ = false;
                return;
            }
        }
    }
    // 没找到，就直接释放内存
    state = cudaFree(ptr);
    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}

std::shared_ptr<DeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

} // namespace base