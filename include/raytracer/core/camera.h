//#pragma once
//#include <glm/glm.hpp>
//
//class Camera {
//public:
//    Camera(){}
//    Camera
//    (
//        int imageWidth,
//        int imageHeight,
//        float focalLength,
//        float viewPortH,
//        glm::vec3 cameraCntr
//    ) : 
//        m_FocalLength(focalLength),
//        m_ViewPortHeight(viewPortH),
//        m_CameraCenter(cameraCntr) 
//    {
//        m_ViewPortWidth = m_ViewPortHeight * (static_cast<float>(imageWidth) / static_cast<float>(imageHeight));
//        // Calculate the vectors across the horizontal and down the vertical viewport edges.
//        auto viewport_u = glm::vec3(m_ViewPortWidth, 0, 0);
//        auto viewport_v = glm::vec3(0, -m_ViewPortHeight, 0);
//
//        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
//        m_PixelDeltaU = viewport_u / static_cast<float>(imageWidth);
//        m_PixelDeltaV = viewport_v / static_cast<float>(imageHeight);
//
//        // Calculate the location of the upper left pixel.
//        auto viewport_upper_left = m_CameraCenter - glm::vec3(0.0f, 0.0f, m_FocalLength) - viewport_u / 2.0f - viewport_v / 2.0f;
//        m_UpperLeftPxl = viewport_upper_left + 0.5f * (m_PixelDeltaU + m_PixelDeltaV);
//    }
//
//    glm::vec3 Center() const 
//    {
//        return m_CameraCenter;
//    }
//
//    glm::vec3 PixelDeltaU() const 
//    {
//        return m_PixelDeltaU;
//    }
//
//    glm::vec3 PixelDeltaV() const
//    {
//        return m_PixelDeltaV;
//    }
//
//    glm::vec3 UpperLeftPixel() const 
//    {
//        return m_UpperLeftPxl;
//    }
//
//private:
//    float m_FocalLength;
//    float m_ViewPortHeight;
//    float m_ViewPortWidth;
//    glm::vec3 m_CameraCenter;
//
//    glm::vec3 m_UpperLeftPxl;
//    glm::vec3 m_PixelDeltaU;
//    glm::vec3 m_PixelDeltaV;
//};