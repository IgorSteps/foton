#include "engine/cuda/InteropBuffer.h"
#include <stdio.h>
#include <glm/glm.hpp>


InteropBuffer::InteropBuffer(PBO* pbo) : _cudaResource(nullptr), _pbo(pbo)
{
	cudaError_t error = cudaGraphicsGLRegisterBuffer(&_cudaResource, _pbo->GetID(), cudaGraphicsMapFlagsWriteDiscard);
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

void InteropBuffer::Update(float width, float height)
{
	// Unregister the PBO from CUDA
	cudaGraphicsUnregisterResource(_cudaResource);

	// Resize the PBO
	_pbo->Update(width, height);

	// Re-register the PBO with CUDA
	cudaGraphicsGLRegisterBuffer(&_cudaResource, _pbo->GetID(), cudaGraphicsMapFlagsWriteDiscard);
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
