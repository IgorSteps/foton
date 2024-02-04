#include <glad/glad.h>
#include "engine/Window.h"
#include "engine/gl/GL.h"
#include <stdexcept>



Window::Window(int width, int height, const std::string& title)
    : _width(width), _height(height), _title(title)
{
    GL::InitiliseGLFW();

    _window = glfwCreateWindow(_width, _height, _title.c_str(), nullptr, nullptr);
    if (!_window) {
        throw std::runtime_error("Failed to create GLFW window");
    }

    GL::InitiliseGLAD(_window);

    glfwSetFramebufferSizeCallback(_window, FramebufferSizeCallback);
}

Window::~Window() {
    glfwDestroyWindow(_window);
    glfwTerminate();
}

void Window::Update() {
    glfwSwapBuffers(_window);
    glfwPollEvents();
}

bool Window::IsClosed() const {
    return glfwWindowShouldClose(_window) != 0;
}

void Window::FramebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}
