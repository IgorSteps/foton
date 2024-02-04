#pragma once
#include <glm/glm.hpp>
#include <glad/glad.h>

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
    glm::mat4 GetViewMatrix();
    void ProcessKeyboard(CameraMovement direction, float deltaTime);
    void ProcessMouseMovement(float xOffset, float yOffset, GLboolean constrainPitch);
    void ProcessMouseScroll(float yOffset);

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