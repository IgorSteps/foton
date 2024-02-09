#pragma once

class CudaBuffer 
{
public:
    CudaBuffer(size_t size);
    ~CudaBuffer();

    void* GetDevicePtr() const;
    size_t GetSize() const;

private:
    void* _devicePtr;
    size_t _size;
};
