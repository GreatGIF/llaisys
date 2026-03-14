#include "../runtime_api.hpp"
#include "../../utils/cuda_check.hpp"

#include <cstdlib>
#include <cstring>

namespace llaisys::device::mx {

namespace runtime_api {
int getDeviceCount() {
    int iDeviceCount = 0;
    LLAISYS_CUDA_CHECK(cudaGetDeviceCount(&iDeviceCount));
    return iDeviceCount;
}

void setDevice(int device_id) {
    LLAISYS_CUDA_CHECK(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    LLAISYS_CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    LLAISYS_CUDA_CHECK(cudaStreamCreate(&stream));
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    LLAISYS_CUDA_CHECK(cudaStreamDestroy((cudaStream_t)stream));
}
void streamSynchronize(llaisysStream_t stream) {
    LLAISYS_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
}

void *mallocDevice(size_t size) {
    void *ptr;
    LLAISYS_CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    LLAISYS_CUDA_CHECK(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr;
    LLAISYS_CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    LLAISYS_CUDA_CHECK(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    LLAISYS_CUDA_CHECK(cudaMemcpy(dst, src, size, (cudaMemcpyKind)kind));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream = 0) {
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(dst, src, size, (cudaMemcpyKind)kind, (cudaStream_t)stream));
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
} // namespace llaisys::device::mx
