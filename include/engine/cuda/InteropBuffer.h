#pragma once
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <engine/gl/PBOBuffer.h>

class InteropBuffer 
{
public:
    InteropBuffer(PBO* pbo);
    ~InteropBuffer();
    void Update(float width, float height);
    void MapCudaResource();
    void UnmapCudaResource();
    void* GetCudaMappedPtr(size_t* size = nullptr);

private:
    cudaGraphicsResource* _cudaResource;
    PBO* _pbo;
};
