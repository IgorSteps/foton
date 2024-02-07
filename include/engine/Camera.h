#pragma once
#include <glm/glm.hpp>
#include <glad/glad.h>
#include <engine/Ray.h>
#include "cuda_runtime.h"

enum CameraMovement 
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

class Camera
{
public:
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch);
    glm::mat4 GetViewMatrix() const;
    float GetZoom() const;
    glm::vec3 GetPosition() const;
    void ProcessKeyboard(CameraMovement direction, float deltaTime);
    void ProcessMouseMovement(float xOffset, float yOffset, GLboolean constrainPitch = true);
    void ProcessMouseScroll(float yOffset);
    __device__ Ray GetRay(float u, float v) const;
private:
    // Camera attributes.
    glm::vec3 _position;
    glm::vec3 _front;
    glm::vec3 _up;
    glm::vec3 _right;
    glm::vec3 _worldUp;

    // Euler angles.
    float _yaw;
    float _pitch;

    // Camera options.
    float _speed;
    float _sensitivity;
    float _zoom;

    void updateCameraVectors();
};