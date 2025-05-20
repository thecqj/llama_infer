#pragma once

#include <cuda_runtime.h>

namespace kernel {

struct CudaConfig {
    cudaStream_t stream = nullptr;

    CudaConfig(cudaStream_t st = nullptr) : stream{st} {}

    ~CudaConfig() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
};

} // namespace kernel