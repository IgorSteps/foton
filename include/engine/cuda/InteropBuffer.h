#pragma once
#include <glad/glad.h>
#include <cuda_gl_interop.h>

class InteropBuffer 
{
public:
    InteropBuffer(GLuint bufferID);
    ~InteropBuffer();

    void MapCudaResource();
    void UnmapCudaResource();
    void* GetCudaMappedPtr(size_t* size = nullptr);

private:
    cudaGraphicsResource* _cudaResource;
    GLuint _bufferID;
};
