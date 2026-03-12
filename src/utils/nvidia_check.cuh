#pragma once

#ifdef ENABLE_NVIDIA_API

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace llaisys::utils {

inline const char *cublas_status_to_str(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}

inline void cuda_check(cudaError_t err, const char *call, const char *file, int line, const char *func) {
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA call failed: " << cudaGetErrorString(err) << " (" << call << ")"
                  << " from " << func << " at " << file << ":" << line << "." << std::endl;
        throw std::runtime_error("CUDA call failed");
    }
}

inline void cublas_check(cublasStatus_t st, const char *call, const char *file, int line, const char *func) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[ERROR] cuBLAS call failed: " << cublas_status_to_str(st) << " (" << call << ")"
                  << " from " << func << " at " << file << ":" << line << "." << std::endl;
        throw std::runtime_error("cuBLAS call failed");
    }
}

} // namespace llaisys::utils

#define LLAISYS_CUDA_CHECK(CALL__) \
    ::llaisys::utils::cuda_check((CALL__), #CALL__, __FILE__, __LINE__, __func__)

#define LLAISYS_CUBLAS_CHECK(CALL__) \
    ::llaisys::utils::cublas_check((CALL__), #CALL__, __FILE__, __LINE__, __func__)

#endif // ENABLE_NVIDIA_API
