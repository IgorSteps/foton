#include <glm/glm.hpp>

// Simplified representation of Camera class to use for calculations on the GPU.
struct CameraData 
{
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    float fov;
    float aspectRatio;
};