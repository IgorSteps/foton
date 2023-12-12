#include <renderer/renderer.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;


int main() 
{
    auto start = high_resolution_clock::now();

    // Image
    Image image = Image();
    Camera camera = Camera(image.Width(), image.Height(), 1.0f, 2.0f, glm::vec3(0, 0, 0));

    // Sphere
    Sphere sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f);

    int numIterations = 10; // Number of iterations
    float totalKernelTime = 0, totalMallocTime = 0, totalMemcpyTime = 0;
    float kernelTime, mallocTime, memcpyTime;
    // Render
   // for (int i = 0; i < numIterations; ++i) {
        Renderer r(image, camera, sphere);
        r.Render(&kernelTime, &mallocTime, &memcpyTime);
        totalKernelTime += kernelTime;
        totalMallocTime += mallocTime;
        totalMemcpyTime += memcpyTime;
    //}
    float avgKernelTime = totalKernelTime / numIterations;
    float avgMallocTime = totalMallocTime / numIterations;
    float avgMemcpyTime = totalMemcpyTime / numIterations;

    std::clog << "Average RenderKernel time: " << avgKernelTime << " ms\n";
    std::clog << "Average cudaMalloc time: " << avgMallocTime << " ms\n";
    std::clog << "Average cudaMemcpy time: " << avgMemcpyTime << " ms\n";

    return 0;
}