#include <engine/Camera.h>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) 
	: 
	_front(glm::vec3(0.0f, 0.0f, -1.0f)),
	_speed(2.5f),
	_sensitivity(0.1f),
	_zoom(45.0f)
{
	_position = position;
	_worldUp = up;
	_yaw = yaw;
	_pitch = pitch;
	updateCameraVectors();
}

glm::mat4 Camera::GetViewMatrix() const
{
	return glm::lookAt(_position, _position + _front, _up);
}

float Camera::GetZoom() const
{
    return _zoom;
}

glm::vec3 Camera::GetPosition() const
{
    return  _position;
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

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (constrainPitch)
    {
        if (_pitch > 89.0f)
            _pitch = 89.0f;
        if (_pitch < -89.0f)
            _pitch = -89.0f;
    }

    // update Front, Right and Up Vectors using the updated Euler angles
    updateCameraVectors();
}

void Camera::ProcessMouseScroll(float yoffset)
{
    _zoom -= (float)yoffset;
    if (_zoom < 1.0f)
        _zoom = 1.0f;
    if (_zoom > 45.0f)
        _zoom = 45.0f;
}

/// <summary>
/// Generate a ray passing through a given pixel on the viewport.
/// </summary>
__device__ Ray Camera::GetRay(float u, float v) const
{
    float aspectRatio = 1200 / (float)800;
    
    // tanFovHalf allows to calculate how far a Ray should diverge from the camera's
    // central direction as it passes through a given pixel on the image plane.
    float tanFovHalf = tan(glm::radians(_zoom / 2.0f)); 

    // Convert from normalised screen coordinates to NDC.
    float ndcX = (2.0f * u) - 1.0f;
    float ndcY = 1.0f - (2.0f * v); // Flip Y axis(OpengGL and GLFW coord systems are inverted)

    // Convert from NDC to camera coordinates.
    float camX = ndcX * aspectRatio * tanFovHalf;
    float camY = ndcY * tanFovHalf;

    // Create a ray in camera space.
    glm::vec3 rayDirCamSpace = glm::normalize(glm::vec3(camX, camY, -1.0f));

    // Convert ray direction from camera space to world space.
    glm::vec3 rayDirWorldSpace = glm::normalize(_front + rayDirCamSpace.x * _right - rayDirCamSpace.y * _up);

    return Ray(_position, rayDirWorldSpace);
}

void Camera::updateCameraVectors()
{
    // calculate the new Front vector
    glm::vec3 front;
    front.x = cos(glm::radians(_yaw)) * cos(glm::radians(_pitch));
    front.y = sin(glm::radians(_pitch));
    front.z = sin(glm::radians(_yaw)) * cos(glm::radians(_pitch));
    _front = glm::normalize(front);
    // also re-calculate the Right and Up vector
    _right = glm::normalize(glm::cross(_front, _worldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    _up = glm::normalize(glm::cross(_right, _front));
}