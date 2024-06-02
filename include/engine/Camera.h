#pragma once
#include <glm/glm.hpp>
#include <glad/glad.h>
#include "cuda_runtime.h"
#include <engine/message/EventQueue.h>

struct CameraData
{
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    float fov;
    float aspectRatio;
};

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

    float GetZoom() const;
    glm::vec3 GetPosition() const;
    glm::vec3 GetUp() const;
    glm::vec3 GetFront() const;
    glm::vec3 GetRight() const;

    void ProcessKeyboard(CameraMovement direction, float deltaTime);
    void ProcessMouseMovement(float xOffset, float yOffset, GLboolean constrainPitch = true);
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

extern EventQueue eventQueue;