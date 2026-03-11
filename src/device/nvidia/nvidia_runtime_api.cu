#include "../runtime_api.hpp"
#include "./common.cuh"

#include <cstdlib>
#include <cstring>

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int iDeviceCount = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&iDeviceCount), __FILE__, __LINE__);
    return iDeviceCount;
}

void setDevice(int device_id) {
    cudaError_t error = ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);
}

void deviceSynchronize() {
    cudaError_t error = ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    cudaError_t error = ErrorCheck(cudaStreamCreate(&stream), __FILE__, __LINE__);
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    cudaError_t error = ErrorCheck(cudaStreamDestroy((cudaStream_t)stream), __FILE__, __LINE__);
}
void streamSynchronize(llaisysStream_t stream) {
    cudaError_t error = ErrorCheck(cudaStreamSynchronize((cudaStream_t)stream), __FILE__, __LINE__);
}

void *mallocDevice(size_t size) {
    void *ptr;
    cudaError_t error = ErrorCheck(cudaMalloc(&ptr, size), __FILE__, __LINE__);
    return ptr;
}

void freeDevice(void *ptr) {
    cudaError_t error = ErrorCheck(cudaFree(ptr), __FILE__, __LINE__);
}

void *mallocHost(size_t size) {
    void *ptr;
    cudaError_t error = ErrorCheck(cudaMallocHost(&ptr, size), __FILE__, __LINE__);
    return ptr;
}

void freeHost(void *ptr) {
    cudaError_t error = ErrorCheck(cudaFreeHost(ptr), __FILE__, __LINE__);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaError_t error = ErrorCheck(cudaMemcpy(dst, src, size, (cudaMemcpyKind)kind), __FILE__, __LINE__);
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream = 0) {
    cudaError_t error = ErrorCheck(cudaMemcpyAsync(dst, src, size, (cudaMemcpyKind)kind, (cudaStream_t)stream), __FILE__, __LINE__);
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
