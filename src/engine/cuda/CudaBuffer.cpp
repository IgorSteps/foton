#include <engine/cuda/CudaBuffer.h>
#include "cuda_runtime.h"


CudaBuffer::CudaBuffer(size_t size) : _size(size), _devicePtr(nullptr)
{}


CudaBuffer::~CudaBuffer() {
    cudaFree(_devicePtr);
}

void* CudaBuffer::GetDevicePtr() const {
    return _devicePtr;
}

size_t CudaBuffer::GetSize() const {
    return _size;
}