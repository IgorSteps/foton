#include <glm/glm.hpp>

struct CameraData 
{
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    float fov;
    float aspectRatio;
};

struct SphereData 
{
    glm::vec3 position;
    float radius;
};

__global__ void renderKernel(glm::vec3* colors, int width, int height, CameraData* camData, SphereData* sphereData);
