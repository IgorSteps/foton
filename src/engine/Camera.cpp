#include <engine/Camera.h>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) 
	: 
	_front(glm::vec3(0.0f, 0.0f, -1.0f)),
	_speed(5.5f),
	_sensitivity(0.1f),
	_zoom(45.0f)
{
	_position = position;
	_worldUp = up;
	_yaw = yaw;
	_pitch = pitch;
	UpdateVectors();
}

float Camera::GetZoom() const
{
    return _zoom;
}

glm::vec3 Camera::GetPosition() const
{
    return  _position;
}

glm::vec3 Camera::GetUp() const
{
    return _up;
}

glm::vec3 Camera::GetFront() const
{
    return _front;
}

glm::vec3 Camera::GetRight() const
{
    return _right;
}

void Camera::ProcessKeyboard(CameraMovement direction, float deltaTime)
{
    float velocity = _speed * deltaTime;
    if (direction == FORWARD)
        _position += _front * velocity;
    if (direction == BACKWARD)
        _position -= _front * velocity;
    if (direction == LEFT)
        _position -= _right * velocity;
    if (direction == RIGHT)
        _position += _right * velocity;
}

void Camera::ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch)
{
    xoffset *= _sensitivity;
    yoffset *= _sensitivity;

    _yaw += xoffset;
    _pitch += yoffset;

    // Makes sure the screen doesn't get flipped if pitch is out of bounds .
    if (constrainPitch)
    {
        if (_pitch > 89.0f)
            _pitch = 89.0f;
        if (_pitch < -89.0f)
            _pitch = -89.0f;
    }

    UpdateVectors();
}

void Camera::UpdateVectors()
{
    // Recalculate the vectors.
    glm::vec3 front;
    front.x = cos(glm::radians(_yaw)) * cos(glm::radians(_pitch));
    front.y = sin(glm::radians(_pitch));
    front.z = sin(glm::radians(_yaw)) * cos(glm::radians(_pitch));
    _front = glm::normalize(front);
    _right = glm::normalize(glm::cross(_front, _worldUp));
    _up = glm::normalize(glm::cross(_right, _front));
}