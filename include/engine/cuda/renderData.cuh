#include <engine/Camera.h>
#include <engine/graphics/SphereSprite.h>

#include <glm/glm.hpp>

struct CameraData {
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    float fov;
    float aspectRatio;

    //CameraData(Camera* cam)
    //    : position(cam->GetPosition()),
    //    front(cam->GetFront()),
    //    up(cam->GetUp()),
    //    right(cam->GetRight()),
    //    fov(cam->GetZoom()),
    //    aspectRatio(1200.0f / 800.0f) {}
};

struct SphereData {
    glm::vec3 position;
    float radius;

    //SphereData(SphereSprite* sphere)
    //    : position(sphere->position),
    //    radius(sphere->GetRadius()) {}
};

__global__ void renderKernel(glm::vec3* colors, int width, int height, CameraData* camData, SphereData* sphereData);
