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

const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 800;

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
    _window = std::make_unique<Window>(SCR_WIDTH, SCR_HEIGHT, "Foton");
    _camera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    _quadSprite = std::make_unique<QuadSprite>("Test quad", 0.5f, 0.5f);
    _sphereSprite = std::make_unique<SphereSprite>("Test sphere", 1.0f, 36, 18);
    _renderer = std::make_unique<Renderer>(_camera.get(), _sphereSprite.get());
    _texture = std::make_unique<Texture>((float)SCR_WIDTH, (float)SCR_HEIGHT);
    
    loadShaders();
    _basicShader->Use();

    _quadSprite->Init();
    _sphereSprite->Init();
    _texture->Init();
}

void Engine::update(float dt)
{
    updateCameraFromEvent(_camera, dt);
}

// Render here
void Engine::draw()
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Generate ray-traced image and upload it as texture to GPU.
    _renderer->Render();
    _texture->ActivateAndBind();
    _texture->Upload(_renderer->image);

    //@TODO: move elsewhere?
    auto textLocation = _basicShader->GetUniformLocation("u_texture");
    glUniform1i(textLocation, 0);

    // Draw the quad.
    _quadSprite->Draw(_basicShader);
}

void Engine::loadShaders()
{
    // Working dir is D/Projects/foton.
    std::string vertexShaderSource = FileIO::ReadFile(".\\src\\engine\\gl\\shaders\\basicVertex.vert");
    std::string fragmentShaderSource = FileIO::ReadFile(".\\src\\engine\\gl\\shaders\\basicFrag.frag");

    _basicShader = std::make_unique<Shader>("Basic");
    _basicShader->Load(vertexShaderSource, fragmentShaderSource);
}

void Engine::updateCameraFromEvent(std::unique_ptr<Camera>& camera, float dt)
{
    Event event;

    while (eventQueue.PollEvent(event))
    {
        switch (event.type)
        {
        case EventType::MoveForward:
            camera->ProcessKeyboard(FORWARD, dt);
            break;
        case EventType::MoveBackward:
            camera->ProcessKeyboard(BACKWARD, dt);
            break;
        case EventType::MoveLeft:
            camera->ProcessKeyboard(LEFT, dt);
            break;
        case EventType::MoveRight:
            camera->ProcessKeyboard(RIGHT, dt);
            break;
        case EventType::LookAround:
            camera->ProcessMouseMovement(event.xoffset, event.yoffset);
            break;
        case EventType::Zoom:
            camera->ProcessMouseScroll(event.yoffset);
            break;
        }
    }
}
