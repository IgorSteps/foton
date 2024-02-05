#include <glad/glad.h>
#include "engine/Window.h"
#include "engine/gl/GL.h"
#include <stdexcept>
#include <engine/message/Event.h>

bool firstMouse = true;

float Window::_lastX;
float Window::_lastY;

Window::Window(int width, int height, const std::string& title)
    : _width(width), _height(height), _title(title)
{
    // Start at 0,0.
    _lastY = _width / 2.0f;
    _lastX = _height / 2.0f;

    GL::InitiliseGLFW();

    _window = glfwCreateWindow(_width, _height, _title.c_str(), nullptr, nullptr);
    if (!_window) {
        throw std::runtime_error("Failed to create GLFW window");
    }

    GL::InitiliseGLAD(_window);
    glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetFramebufferSizeCallback(_window, FramebufferSizeCallback);
    glfwSetCursorPosCallback(_window, MouseCallback);
    glfwSetScrollCallback(_window, ScrollCallback);
}

Window::~Window() 
{
    glfwDestroyWindow(_window);
    glfwTerminate();
}

void Window::Update() 
{
    ProcessInput(_window);
    glfwSwapBuffers(_window);
    glfwPollEvents();
}

bool Window::IsClosed() const 
{
    return glfwWindowShouldClose(_window) != 0;
}

void Window::FramebufferSizeCallback(GLFWwindow* window, int width, int height) 
{
    glViewport(0, 0, width, height);
}

void Window::MouseCallback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        _lastX = xpos;
        _lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - _lastX;
    float yoffset = _lastY - ypos; // reversed since y-coordinates go from bottom to top

    _lastX = xpos;
    _lastY = ypos;

    eventQueue.PostEvent(Event(EventType::LookAround, xoffset, yoffset));
}

void Window::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    eventQueue.PostEvent(Event(EventType::Zoom, yoffset));
}

void Window::ProcessInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        eventQueue.PostEvent(Event(EventType::MoveForward));
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        eventQueue.PostEvent(Event(EventType::MoveBackward));
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        eventQueue.PostEvent(Event(EventType::MoveLeft));
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        eventQueue.PostEvent(Event(EventType::MoveRight));
}

