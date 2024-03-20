#include "engine/Engine.h"
#include <engine/io/File.h>
#include <filesystem>
#include "glm/gtc/matrix_transform.hpp"
#include "glm/ext.hpp"
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::seconds;

const float SCR_WIDTH = 1200.0f;
const float SCR_HEIGHT = 800.0f;

GLenum glCheckError_(const char* file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
        case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
        case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
        case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
        case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
        case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
        case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
        case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
    }
    return errorCode;
}
#define glCheckError() glCheckError_(__FILE__, __LINE__) 


Engine::Engine()
{
    init();
}

void Engine::run()
{
    auto lastTime = Clock::now();

    try 
    {
        while (!_window->IsClosed())
        {
            auto currentTime = Clock::now();
            float dt = duration_cast<duration<float>>(currentTime - lastTime).count();
            lastTime = currentTime;

            // @TODO: Move elsewhere.
            // FPS:
            _timeSinceLastFPSUpdate += dt;
            ++_frameCount;

            // Check if it's time to update the FPS display
            if (_timeSinceLastFPSUpdate >= _fpsUpdateInterval)
            {
                _lastFPS = _frameCount / _timeSinceLastFPSUpdate;
                _frameCount = 0;
                _timeSinceLastFPSUpdate = 0.0f;

                // Display FPS in window title.
                std::string windowTitle = "Foton - FPS: " + std::to_string(_lastFPS);
                _window->SetTitle(windowTitle);
            }

            update(dt);
            draw();
            _window->Update();
        }
    }
    catch (const std::exception& e) 
    {
        std::cerr << e.what() << std::endl;
    }
}

void Engine::init()
{
    // @TODO: Set aspect ratio based on viewport width & height.
    
    // Make window and initilise OpenGL context.
    _window = std::make_unique<Window>(SCR_WIDTH, SCR_HEIGHT, "Foton");

    // Load shaders.
    loadShaders();
    _shader->Use();

    // Initilise world.
    h_Camera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    _rayTracedImage = std::make_unique<RayTracedImage>(SCR_WIDTH, SCR_HEIGHT);
    _rayTracedImage->Init();
    _interopBuffer = std::make_unique<InteropBuffer>(_rayTracedImage->GetPBOID());

    // Spheres.
    Sphere mainSphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f, glm::vec3(1.0f, 0.5f, 0.31f), false);
    _spheres.push_back(mainSphere);
   /* Sphere mainSphere1(glm::vec3(1.0f, 2.0f, -2.0f), 0.5f, glm::vec3(1.0f, 0.5f, 0.31f), false);
    _spheres.push_back(mainSphere1);  
    Sphere mainSphere2(glm::vec3(1.0f, -2.0f, -2.0f), 0.5f, glm::vec3(1.0f, 0.5f, 0.31f), false);
    _spheres.push_back(mainSphere2); 
    Sphere mainSphere3(glm::vec3(-1.0f, -2.0f, -3.0f), 0.5f, glm::vec3(1.0f, 0.5f, 0.31f), false);
    _spheres.push_back(mainSphere3);  
    Sphere mainSphere4(glm::vec3(2.0f, 2.0f, -2.0f), 0.5f, glm::vec3(1.0f, 0.5f, 0.31f), false);
    _spheres.push_back(mainSphere4);*/
    Sphere lightSphere(glm::vec3(1.0f, 1.0f, 0.5f), 0.25f, glm::vec3(1.0f), true);
    _spheres.push_back(lightSphere);

    // Lights.
    light = new Light(glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(1.0f), 1.5);
    
    // Make renderer.
    _renderer = std::make_unique<Renderer>(h_Camera.get(), light, _spheres);
}

void Engine::update(float dt)
{
    // Update Camera data on CPU.
    h_Camera->Update(dt);

    // Update PBO with CUDA.
    _renderer->Update(_interopBuffer);

    // Update Camera data on GPU.
    _renderer->UpdateCameraData();

    // Update light.
    light->Update(dt);
    _renderer->UpdateLightData();

    // Update image: buffers, textures...
    _rayTracedImage->Update();
}

void Engine::draw()
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw final image.
    _rayTracedImage->Draw(_shader);

    glCheckError();
}

void Engine::loadShaders()
{
    // Working dir is D/Projects/foton.
    std::string vertexShaderSource = FileIO::ReadFile(".\\src\\engine\\gl\\shaders\\basicVertex.vert");
    std::string fragmentShaderSource = FileIO::ReadFile(".\\src\\engine\\gl\\shaders\\basicFrag.frag");

    _shader = std::make_unique<Shader>("Basic");
    _shader->Load(vertexShaderSource, fragmentShaderSource);
}
