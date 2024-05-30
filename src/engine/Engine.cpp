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

float OLD_SCR_WIDTH = 1200.0f;
float OLD_SCR_HEIGHT = 800.0f;
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
    // Make window and initilise OpenGL context.
    _window = std::make_unique<Window>(SCR_WIDTH, SCR_HEIGHT, "Foton");

    // Shaders:
    loadShaders();
    _shader->Use();

    // PBO:
    _pbo = new PBO(SCR_WIDTH, SCR_HEIGHT);

    // Interop buffer:
    _interopBuffer = std::make_unique<InteropBuffer>(_pbo);

    // Ray-Traced image
    _rayTracedImage = std::make_unique<RayTracedImage>(_pbo, SCR_WIDTH, SCR_HEIGHT);
    _rayTracedImage->Init();

    // ------------------------------------
    //              SETUP WORLD
    // ------------------------------------
    // Camera:
    _camera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);

    // Spheres:
    Sphere groundSphere(glm::vec3(0.0f, -510.0f, -1.0f), 500.0f, glm::vec3(0.96f, 0.96f, 0.86f), false);
    _spheres.push_back(groundSphere);
    
    Sphere mainSphere(glm::vec3(0.0f, 0.0f, -1.0f), 1.0f, glm::vec3(1.0f, 0.5f, 0.31f), false);
    _spheres.push_back(mainSphere);
    Sphere lightSphere(glm::vec3(3.0f, 3.0f, -0.5f), 0.3f, glm::vec3(1.0f), true);
    _spheres.push_back(lightSphere);

   /* Sphere mainSphere2(glm::vec3(1.0f, 0.0f, 1.0f), 0.7f, glm::vec3(0.3f, 0.5f, 0.31f), false);
    _spheres.push_back(mainSphere2);
    Sphere mainSphere3(glm::vec3(5.0f, 0.0f, -3.0f), 2.0f, glm::vec3(1.0f, 0.5f, 1.0f), false);
    _spheres.push_back(mainSphere3);*/
    
    // Adding more spheres here:
    //Sphere sphere1(glm::vec3(-4.0f, 1.0f, -4.0f), 1.2f, glm::vec3(0.7f, 0.2f, 0.2f), false);
    //_spheres.push_back(sphere1);

    //Sphere sphere2(glm::vec3(5.0f, 0.5f, 6.0f), 0.8f, glm::vec3(0.2f, 0.7f, 0.3f), false);
    //_spheres.push_back(sphere2);

    //Sphere sphere3(glm::vec3(-6.0f, -1.0f, 3.0f), 1.5f, glm::vec3(0.2f, 0.2f, 0.8f), false);
    //_spheres.push_back(sphere3);

    //Sphere sphere4(glm::vec3(0.0f, 2.0f, -8.0f), 1.0f, glm::vec3(0.8f, 0.8f, 0.0f), false);
    //_spheres.push_back(sphere4);

    //Sphere sphere5(glm::vec3(8.0f, -0.5f, 0.0f), 1.3f, glm::vec3(0.0f, 0.5f, 0.5f), false);
    //_spheres.push_back(sphere5);

    //Sphere sphere6(glm::vec3(-3.0f, -0.5f, -6.0f), 0.9f, glm::vec3(0.5f, 0.0f, 0.5f), false);
    //_spheres.push_back(sphere6);

    //Sphere sphere7(glm::vec3(6.0f, 1.5f, -2.0f), 1.1f, glm::vec3(0.1f, 0.4f, 0.7f), false);
    //_spheres.push_back(sphere7);

    //Sphere sphere8(glm::vec3(-5.0f, -2.0f, 4.0f), 1.4f, glm::vec3(0.9f, 0.6f, 0.0f), false);
    //_spheres.push_back(sphere8);

    //Sphere sphere9(glm::vec3(4.0f, -1.5f, 6.0f), 0.7f, glm::vec3(0.3f, 0.3f, 0.3f), false);
    //_spheres.push_back(sphere9);

    //Sphere sphere10(glm::vec3(1.0f, 3.0f, -5.0f), 1.0f, glm::vec3(0.0f, 0.8f, 0.4f), false);
    //_spheres.push_back(sphere10);

    //Sphere sphere11(glm::vec3(-7.0f, 0.0f, 1.0f), 1.3f, glm::vec3(0.6f, 0.1f, 0.2f), false);
    //_spheres.push_back(sphere11);

    //Sphere sphere12(glm::vec3(2.0f, -2.5f, -1.0f), 1.2f, glm::vec3(0.4f, 0.7f, 0.2f), false);
    //_spheres.push_back(sphere12);

    //Sphere sphere13(glm::vec3(-1.0f, 1.5f, 5.0f), 0.6f, glm::vec3(0.3f, 0.6f, 0.8f), false);
    //_spheres.push_back(sphere13);

    //Sphere sphere14(glm::vec3(7.0f, -1.0f, -3.0f), 1.4f, glm::vec3(0.5f, 0.3f, 0.7f), false);
    //_spheres.push_back(sphere14);

    //Sphere sphere15(glm::vec3(-4.0f, 2.5f, -1.0f), 1.0f, glm::vec3(0.2f, 0.8f, 0.8f), false);
    //_spheres.push_back(sphere15);

    //Sphere sphere16(glm::vec3(0.0f, 0.0f, 6.0f), 1.2f, glm::vec3(0.9f, 0.3f, 0.7f), false);
    //_spheres.push_back(sphere16);

    //Sphere sphere17(glm::vec3(5.0f, -3.0f, 0.0f), 1.1f, glm::vec3(0.1f, 0.6f, 0.4f), false);
    //_spheres.push_back(sphere17);

    //Sphere sphere18(glm::vec3(-3.0f, 2.0f, 3.0f), 0.8f, glm::vec3(0.4f, 0.4f, 0.9f), false);
    //_spheres.push_back(sphere18);

    //Sphere sphere19(glm::vec3(2.0f, 3.0f, -6.0f), 1.3f, glm::vec3(0.7f, 0.7f, 0.1f), false);
    //_spheres.push_back(sphere19);

    //Sphere sphere20(glm::vec3(-6.0f, -1.5f, -4.0f), 0.9f, glm::vec3(0.5f, 0.2f, 0.6f), false);
    //_spheres.push_back(sphere20);

    //Sphere sphere21(glm::vec3(4.0f, 1.0f, -4.0f), 1.0f, glm::vec3(0.6f, 0.3f, 0.8f), false);
    //_spheres.push_back(sphere21);

    //Sphere sphere22(glm::vec3(-1.0f, -3.0f, 1.0f), 1.4f, glm::vec3(0.8f, 0.2f, 0.4f), false);
    //_spheres.push_back(sphere22);

    //Sphere sphere23(glm::vec3(6.0f, -0.5f, 2.0f), 0.7f, glm::vec3(0.2f, 0.5f, 0.9f), false);
    //_spheres.push_back(sphere23);

    //Sphere sphere24(glm::vec3(-2.0f, 3.5f, -1.0f), 1.1f, glm::vec3(0.3f, 0.8f, 0.3f), false);
    //_spheres.push_back(sphere24);

    //Sphere sphere25(glm::vec3(3.0f, -2.0f, -7.0f), 1.2f, glm::vec3(0.9f, 0.4f, 0.1f), false);
    //_spheres.push_back(sphere25);

    //Sphere sphere26(glm::vec3(-5.0f, 0.5f, 5.0f), 0.8f, glm::vec3(0.4f, 0.6f, 0.6f), false);
    //_spheres.push_back(sphere26);

    //Sphere sphere27(glm::vec3(1.0f, -2.5f, 3.0f), 1.0f, glm::vec3(0.7f, 0.5f, 0.5f), false);
    //_spheres.push_back(sphere27);

    //Sphere sphere28(glm::vec3(4.0f, -1.0f, -8.0f), 1.3f, glm::vec3(0.6f, 0.2f, 0.7f), false);
    //_spheres.push_back(sphere28);

    //Sphere sphere29(glm::vec3(-6.0f, 1.0f, 0.0f), 1.4f, glm::vec3(0.3f, 0.9f, 0.3f), false);
    //_spheres.push_back(sphere29);

    //Sphere sphere30(glm::vec3(2.0f, -3.0f, 4.0f), 0.7f, glm::vec3(0.5f, 0.4f, 0.8f), false);
    //_spheres.push_back(sphere30);

    //Sphere sphere31(glm::vec3(0.0f, 2.0f, -3.0f), 1.2f, glm::vec3(0.8f, 0.3f, 0.9f), false);
    //_spheres.push_back(sphere31);

    //Sphere sphere32(glm::vec3(-3.0f, -1.0f, 5.0f), 0.9f, glm::vec3(0.3f, 0.4f, 0.7f), false);
    //_spheres.push_back(sphere32);

    //Sphere sphere33(glm::vec3(7.0f, -2.0f, -2.0f), 1.1f, glm::vec3(0.9f, 0.2f, 0.6f), false);
    //_spheres.push_back(sphere33);

    //Sphere sphere34(glm::vec3(-4.0f, 1.5f, 6.0f), 1.0f, glm::vec3(0.7f, 0.8f, 0.2f), false);
    //_spheres.push_back(sphere34);

    //Sphere sphere35(glm::vec3(3.0f, -0.5f, -5.0f), 1.3f, glm::vec3(0.4f, 0.5f, 0.9f), false);
    //_spheres.push_back(sphere35);

    // Light:
    light = new Light(glm::vec3(3.0f, 3.0f, -0.5f), glm::vec3(1.0f), 1.5);
    
    // Create Render system:
    _renderer = std::make_unique<Renderer>(_camera.get(), light, _spheres);

}

void Engine::update(float dt)
{
    processQueue(dt);

    // If resized window:
    // NOTE: this fails when the window is resized on the fly...
    /*if (SCR_WIDTH != OLD_SCR_WIDTH && SCR_HEIGHT != OLD_SCR_HEIGHT) 
    {
         Update InteropBuffer with resized PBO.
        _interopBuffer->Update(SCR_WIDTH, SCR_HEIGHT);
    }*/

    // Update InteropBuffer with resized PBO.
    _interopBuffer->Update(SCR_WIDTH, SCR_HEIGHT);

    // Update PBO data with CUDA.
    _renderer->Update(SCR_WIDTH, SCR_HEIGHT, _interopBuffer);

    // Update Camera data on GPU.
    _renderer->UpdateCameraData(SCR_WIDTH, SCR_HEIGHT);

    // Update ray traced image: texture.
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
            // Setup old screen dimensions.
            OLD_SCR_WIDTH = SCR_WIDTH;
            OLD_SCR_HEIGHT = SCR_HEIGHT;

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