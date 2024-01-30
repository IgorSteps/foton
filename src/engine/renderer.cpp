#include <engine/renderer.h>
#include <iostream>
#include <glm/glm.hpp>
#include <utils/colour.h>


void Renderer::Render()
{
    int imageHeight = Renderer::m_Img.Height();
    int imageWidth = Renderer::m_Img.Width();

    std::cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

    for (int j = 0; j < imageHeight; ++j)
    {
        std::clog << "\rScanlines remaining: " << (imageHeight - j) << ' ' << std::flush;
        for (int i = 0; i < imageWidth; ++i)
        {
            glm::vec3 pixelCenter = m_Camera.UpperLeftPixel() + (static_cast<float>(i) * m_Camera.PixelDeltaU()) + (static_cast<float>(j) * m_Camera.PixelDeltaV());
            auto rayDirection = pixelCenter - m_Camera.Center();
            Ray r(m_Camera.Center(), rayDirection);

            glm::vec3 pixelColour = CalculateRayColour(r);
            WriteColour(std::cout, pixelColour);
        }
    }
    std::clog << "\rDone.                 \n";
}

// Linearly blend white and blue depending on the height of the y
// coordinate after scaling the ray direction to unit length(so −1.0 < y < 1.0)
glm::vec3 Renderer::CalculateRayColour(const Ray& r) 
{
    // Colour our sphere red if hit by a ray.
    if(m_Sphere.IsHit(r))
    {
        return glm::vec3(1, 0, 0);
    }

    glm::vec3 unitDirection = glm::normalize(r.Direction());
    auto a = 0.5f * (unitDirection.y + 1.0f);
    
    // blendedValue = (1−a)*startValue+a*endValue
    return (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
}
