#include <glog/logging.h>

#include "kernels_interface.h"
#include "cpu/add_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/embedding_kernel.h"
#include "cpu/mha_kernel.h"
#include "cuda/add_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
#include "cuda/matmul_kernel.cuh"
#include "cuda/embedding_kernel.cuh"
#include "cuda/mha_kernel.cuh"

namespace kernel {

AddKernel get_add_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return add_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceGPU) {
        return add_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return rmsnorm_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceGPU) {
        return rmsnorm_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
        return nullptr;
    }
}

MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return matmul_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceGPU) {
        return matmul_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a matmul kernel.";
        return nullptr;
    }
}

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceGPU) {
        return matmul_kernel_cu_qint8;
    } else {
        LOG(FATAL) << "Unknown device type for get a matmul kernel.";
        return nullptr;
    }
}

EmbeddingKernel get_embedding_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return embedding_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceGPU) {
        return embedding_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a matmul kernel.";
        return nullptr;
    }
}

MHAKernel get_mha_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return mha_kernel;
    } else if (device_type == base::DeviceType::kDeviceGPU) {
        return mha_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a matmul kernel.";
        return nullptr;
    }
}

} // namespace kernel