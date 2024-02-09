#include "engine/cuda/InteropBuffer.h"
#include <stdio.h>

InteropBuffer::InteropBuffer(GLuint bufferID) : _cudaResource(nullptr), _bufferID(bufferID)
{
	cudaError_t error = cudaGraphicsGLRegisterBuffer(&_cudaResource, _bufferID, cudaGraphicsMapFlagsWriteDiscard);
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: CUDA failed to register OpenGL PBO buffer: %s\n", cudaGetErrorString(error));
	}
}

InteropBuffer::~InteropBuffer()
{
	cudaError_t error = cudaGraphicsUnregisterResource(_cudaResource);
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: CUDA failed to unregister OpenGL buffer: %s\n", cudaGetErrorString(error));
	}
}

void InteropBuffer::MapCudaResource()
{
	cudaError_t error = cudaGraphicsMapResources(1, &_cudaResource, nullptr);
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: CUDA failed to map CUDA resource: %s\n", cudaGetErrorString(error));
	}
}

void InteropBuffer::UnmapCudaResource()
{
	cudaError_t error = cudaGraphicsUnmapResources(1, &_cudaResource, nullptr);
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: CUDA failed to unmap CUDA resource: %s\n", cudaGetErrorString(error));
	}
}

void* InteropBuffer::GetCudaMappedPtr(size_t* size)
{
	void* ptr = nullptr;
	size_t mappedSize = 0;
	cudaError_t error = cudaGraphicsResourceGetMappedPointer(&ptr, &mappedSize, _cudaResource);
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: CUDA failed to get mapped pointer: %s\n", cudaGetErrorString(error));
	}
	if (size) *size = mappedSize;
	return ptr;
}
