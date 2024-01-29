#include "engine/Engine.h"
#include <engine/File.h>
#include <filesystem>




Engine::Engine()
{
    init();
}

void Engine::run()
{
    try {
        while (!window->IsClosed()) {
            float dt = 2;// TODO: calculate delta time
            update(dt);
            draw();
            window->Update();
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void Engine::init()
{
    window = std::make_unique<Window>(800, 600, "Foton");

    // Define vertex data for a triangle
    std::vector<float> vertices = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };

    std::vector<unsigned int> indices = {
        0, 1, 2
    };

    // Create a GLBuffer object
    buffer = std::make_unique<GLBuffer>(GL_FLOAT, GL_TRIANGLES);

    // Define attribute information
    AttributeInfo positionAttrib(0, 3, 0);
    buffer->AddAttributeLocation(positionAttrib);

    // Set vertex and element data
    buffer->SetVertexData(vertices);
    buffer->SetElementData(indices);

    // Upload data to the GPU
    buffer->UploadData();
    buffer->Unbind();

    loadShaders();
    _basicShader->Use();
}

void Engine::update(float dt)
{

}

void Engine::draw()
{
    // Render here
    buffer->Bind(false);
    buffer->Draw();
    buffer->Unbind();
}

void Engine::loadShaders()
{
    // Working dir is D/Projects/foton.
    std::string vertexShaderSource = FileIO::ReadFile(".\\src\\engine\\gl\\shaders\\basicVertex.vert");
    std::string fragmentShaderSource = FileIO::ReadFile(".\\src\\engine\\gl\\shaders\\basicFrag.frag");

    _basicShader = std::make_unique<Shader>("Basic");
    _basicShader->Load(vertexShaderSource, fragmentShaderSource);

}
