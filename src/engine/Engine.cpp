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

            // @TODO: Move elsewhere.
            // FPS:
            _timeSinceLastFPSUpdate += dt;
            ++_frameCount;

            // Check if it's time to update the FPS display
            if (_timeSinceLastFPSUpdate >= _fpsUpdateInterval) {
                _lastFPS = _frameCount / _timeSinceLastFPSUpdate; // Calculate FPS
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

    _pbo = std::make_unique<PBO>(SCR_WIDTH, SCR_HEIGHT);
    _interopBuffer = std::make_unique<InteropBuffer>(_pbo->getID());
}

void Engine::update(float dt)
{
    updateCameraFromEvent(_camera, dt);
}

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


// Render here
void Engine::draw()
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    _interopBuffer->MapCudaResource();

    size_t size;
    void* cudaPtr = _interopBuffer->GetCudaMappedPtr(&size); 

    _renderer->UpdateCameraData();
    _renderer->UpdateSphereData();

    // Update the PBO data via cudaPtr.
    _renderer->RenderUsingCUDA(cudaPtr);

    _interopBuffer->UnmapCudaResource();
   
    // Update texture with PBO data.
    _pbo->bind();
    _texture->ActivateAndBind();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1200, 800, GL_RGB, GL_FLOAT, nullptr);
    
    _texture->Draw(_basicShader);
    _quadSprite->Draw(_basicShader);


    _texture->Unbind();
    _pbo->unbind();

    glCheckError();
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
