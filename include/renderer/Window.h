#pragma once
#include <string>
#include <GLFW/glfw3.h>

class Window
{
public: 
	Window(int width, int height, const std::string& title);
	~Window();
    void Update();
    bool IsClosed() const;
    void ResizeCallback(GLFWwindow* window, int width, int height);
private:
    GLFWwindow* _window;
    int _width, _height;
    std::string _title;

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};