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
    std::cout << "All initlised, let's go" << std::endl;
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

    // Populate with Spheres:
    // For the Phong Illumination demo:
    Sphere groundSphere(glm::vec3(0.0f, -501.0f, -1.0f), 500.0f, glm::vec3(0.96f, 0.96f, 0.86f), false);
    _spheres.push_back(groundSphere);
    Sphere lightSphere(glm::vec3(3.0f, 3.0f, -0.5f), 0.3f, glm::vec3(1.0f), true);
    _spheres.push_back(lightSphere);
    Sphere mainSphere(glm::vec3(0.0f, 0.0f, -1.0f), 1.0f, glm::vec3(1.0f, 0.5f, 0.31f), false);
    _spheres.push_back(mainSphere);

    // For the Grid demo:
    // Populate(80, 5);
    // PopulateNonUniform(160);

    // Light:
    _light = new Light(glm::vec3(3.0f, 3.0f, -0.5f), glm::vec3(1.0f), 1.5);

    // Grid:
    _grid = new Grid(_spheres);
    
    // Create Render:
    _renderer = std::make_unique<Renderer>(_camera.get(), _light, _spheres, _grid);
}

void Engine::update(float dt)
{
    processQueue(dt);

    // Update InteropBuffer with resized PBO.
    _interopBuffer->Update(SCR_WIDTH, SCR_HEIGHT);
   
    // Uncomment if you want a moving light
    /*_light->Update(dt);
    _renderer->UpdateLightData();*/

    // Choose Kernel to run:
    _renderer->UpdatePhong(SCR_WIDTH, SCR_HEIGHT, _interopBuffer);
    //_renderer->UpdateSimple(SCR_WIDTH, SCR_HEIGHT, _interopBuffer);
    //_renderer->UpdateGrid(SCR_WIDTH, SCR_HEIGHT, _interopBuffer);

    // Update Camera data on GPU.
    _renderer->UpdateCameraData(SCR_WIDTH, SCR_HEIGHT);

    // Update ray-traced image's texture.
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

void Engine::Populate(int numSpheres, int spheresPerRow)
{
    int numRows = (numSpheres + spheresPerRow - 1) / spheresPerRow;
    int spheresPlaced = 0;

    for (int row = 0; row < numRows; ++row)
    {
        for (int i = 0; i < spheresPerRow; ++i)
        {
            if (spheresPlaced >= numSpheres) return;

            float x = (i - spheresPerRow / 2) * 3;
            float z = -1.0f - row * 3;
            glm::vec3 position(x, 0.0f, z);
            glm::vec3 colour(0.5f + 0.5f * (i % 2), 0.5f * (row % 2), 0.5f + 0.5f * ((i + row) % 2));
            _spheres.push_back(Sphere(position, 1.0f, colour, false));

            ++spheresPlaced;
        }
    }
}

void Engine::PopulateNonUniform(int numSpheres)
{
    int spheresPlaced = 0;
    struct SphereCluster {
        glm::vec3 center;
        int count;
        float spread;
    };
    std::vector<SphereCluster> clusters = {
        {glm::vec3(-10.0f, 0.0f, -10.0f), numSpheres / 3, 2.0f},
        {glm::vec3(10.0f, 0.0f, -10.0f), numSpheres / 3, 2.0f},
        {glm::vec3(0.0f, 0.0f, 10.0f), numSpheres / 3, 5.0f}
    };

    for (const SphereCluster& c : clusters)
    {
        for (int i = 0; i < c.count; ++i)
        {
            if (spheresPlaced >= numSpheres) return;

            // Random position.
            float x = c.center.x + ((rand() % 1000) / 500.0f - 1.0f) * c.spread;
            float y = c.center.y;
            float z = c.center.z + ((rand() % 1000) / 500.0f - 1.0f) * c.spread;
            glm::vec3 position(x, y, z);
            glm::vec3 colour(0.5f + 0.5f * (i % 2), 0.5f * (c.count % 2), 0.5f + 0.5f * ((i + c.count) % 2));
            _spheres.push_back(Sphere(position, 1.0f, colour, false));

            ++spheresPlaced;
        }
    }
}
