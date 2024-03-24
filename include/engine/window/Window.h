#pragma once
#include <string>
#include <GLFW/glfw3.h>
#include <memory>
#include <engine/message/EventQueue.h>

class Window
{
public: 
	Window(int width, int height, const std::string& title);
	~Window();
    void Update();
    bool IsClosed() const;
    void SetTitle(const std::string& title);
    static void FramebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void MouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void ProcessInput(GLFWwindow* window);

private:
    GLFWwindow* _window;
    int _width, _height;
    std::string _title;
    
    // Mouse coords at the end of each frame.
    static float _lastX;
    static float _lastY;
};

extern EventQueue eventQueue;