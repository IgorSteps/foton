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

float SCR_WIDTH = 1200.0f;
float SCR_HEIGHT = 800.0f;

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
    _camera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);

    _pbo = new PBO(SCR_WIDTH, SCR_HEIGHT);
    _interopBuffer = std::make_unique<InteropBuffer>(_pbo);

    _rayTracedImage = std::make_unique<RayTracedImage>(_pbo, SCR_WIDTH, SCR_HEIGHT);
    _rayTracedImage->Init();


    // Ground
    Ground ground(glm::vec3(0.0, -1.0f, 0.0), glm::vec3(0.0f, 1.0f, 0.0f));

    // Spheres.
    Sphere mainSphere(glm::vec3(0.0f, 0.0f, -1.0f), 1.0f, glm::vec3(1.0f, 0.5f, 0.31f), false);
    _spheres.push_back(mainSphere);

    Sphere lightSphere(glm::vec3(3.0f, 3.0f, -0.5f), 0.3f, glm::vec3(1.0f), true);
    _spheres.push_back(lightSphere);

    // Lights.
    light = new Light(glm::vec3(3.0f, 3.0f, -0.5f), glm::vec3(1.0f), 1.5);
    
    // Make renderer.
    _renderer = std::make_unique<Renderer>(ground, _camera.get(), light, _spheres);

    //glCheckError();

}

void Engine::update(float dt)
{
    processQueue(dt);

    // Update InteropBuffer with resized PBO.
    _interopBuffer->Update(SCR_WIDTH, SCR_HEIGHT);

    // Update PBO data with CUDA.
    _renderer->Update(SCR_WIDTH, SCR_HEIGHT, _interopBuffer);

    // Update Camera data on GPU.
    _renderer->UpdateCameraData(SCR_WIDTH, SCR_HEIGHT);

    // Update light.
    light->Update(dt);
    _renderer->UpdateLightData();

    // Update image: buffers, textures...
    _rayTracedImage->Update(SCR_WIDTH, SCR_HEIGHT);
}

void Engine::draw()
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw final image.
    _rayTracedImage->Draw(_shader);
}

void Engine::loadShaders()
{
    // Working dir is D/Projects/foton.
    std::string vertexShaderSource = FileIO::ReadFile(".\\src\\engine\\gl\\shaders\\basicVertex.vert");
    std::string fragmentShaderSource = FileIO::ReadFile(".\\src\\engine\\gl\\shaders\\basicFrag.frag");

    _shader = std::make_unique<Shader>("Basic");
    _shader->Load(vertexShaderSource, fragmentShaderSource);
}

void  Engine::processQueue(float dt)
{
    Event event;

    while (eventQueue.PollEvent(event))
    {
        switch (event.type)
        {
        case EventType::WindowResize:
            //printf("Width %i, height %i", event.width, event.height);

            SCR_WIDTH = event.width;
            SCR_HEIGHT = event.height;
            break;
        case EventType::MoveForward:
            _camera->ProcessKeyboard(FORWARD, dt);
            break;
        case EventType::MoveBackward:
            _camera->ProcessKeyboard(BACKWARD, dt);
            break;
        case EventType::MoveLeft:
            _camera->ProcessKeyboard(LEFT, dt);
            break;
        case EventType::MoveRight:
            _camera->ProcessKeyboard(RIGHT, dt);
            break;
        case EventType::LookAround:
            _camera->ProcessMouseMovement(event.xoffset, event.yoffset);
            break;
        }
    }
}