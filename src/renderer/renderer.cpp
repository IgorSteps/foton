#include <renderer/renderer.h>
#include <iostream>
#include <glm/glm.hpp>
#include <utils/colour.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void Renderer::Render()
{
    int imageHeight = Renderer::m_Img.Height();
    int imageWidth = Renderer::m_Img.Width();

    glm::vec3* d_Output;
    cudaMalloc(&d_Output, imageHeight * imageWidth * sizeof(glm::vec3));

    Render(d_Output, imageWidth, imageHeight, m_Camera, m_Sphere);
  
    glm::vec3* h_Output = new glm::vec3[imageWidth * imageHeight];
    cudaMemcpy(h_Output, d_Output, imageWidth * imageHeight * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    std::cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";
    for (int j = 0; j < imageHeight; ++j) 
    {
        std::clog << "\rScanlines remaining: " << (imageHeight - j) << ' ' << std::flush;
        for (int i = 0; i < imageWidth; ++i) 
        {
            WriteColour(std::cout, h_Output[j * imageWidth + i]);
        }
    }

    delete[] h_Output;
    cudaFree(d_Output);

    std::clog << "Done\n";
}

// Linearly blend white and blue depending on the height of the y
// coordinate after scaling the ray direction to unit length(so −1.0 < y < 1.0)
//glm::vec3 Renderer::CalculateRayColour(const Ray& r) 
//{
//    // Colour our sphere red if hit by a ray.
//    if(m_Sphere.IsHit(r))
//    {
//        return glm::vec3(1, 0, 0);
//    }
//
//    glm::vec3 unitDirection = glm::normalize(r.Direction());
//    auto a = 0.5f * (unitDirection.y + 1.0f);
//    
//    // blendedValue = (1−a)*startValue+a*endValue
//    return (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
//}
