#include <glad/glad.h>
#include "engine/gl/GL.h"

#include <stdexcept>

const int OPENGL_MAJOR_V = 4;
const int OPENGL_MINOR_V = 5;

void GL::InitiliseGLAD(GLFWwindow* window) 
{
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLAD");
    }
    glEnable(GL_DEPTH_TEST);
}

void GL::InitiliseGLFW()
{
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OPENGL_MAJOR_V);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OPENGL_MINOR_V);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}
