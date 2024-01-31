#include "engine/Engine.h"
#include <engine/io/File.h>
#include <filesystem>
#include "glm/gtc/matrix_transform.hpp"
#include "glm/ext.hpp"

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

    loadShaders();
    _basicShader->Use();

    // Init.
    _projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
    _sprite = std::make_unique<Sprite>("Test sprite", 0.5f, 0.5f);
    _sprite->Init();
}

void Engine::update(float dt)
{

}

void Engine::draw()
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Render here
    auto colorLocation = _basicShader->GetUniformLocation("myColor");
    glUniform4f(colorLocation, 1.0f, 0.5f, 0.0f, 1.0f);

    // model
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    auto modelLocation = _basicShader->GetUniformLocation("u_model");
    glUniformMatrix4fv(modelLocation, 1, GL_FALSE, glm::value_ptr(model));

    //view
    glm::mat4 view = glm::mat4(1.0f);
    view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
    auto viewLocation = _basicShader->GetUniformLocation("u_view");
    glUniformMatrix4fv(viewLocation, 1, GL_FALSE, glm::value_ptr(view));

    //// projection
    auto projectionLocation = _basicShader->GetUniformLocation("u_projection");
    glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, glm::value_ptr(_projection));



    _sprite->Draw();
}

void Engine::loadShaders()
{
    // Working dir is D/Projects/foton.
    std::string vertexShaderSource = FileIO::ReadFile(".\\src\\engine\\gl\\shaders\\basicVertex.vert");
    std::string fragmentShaderSource = FileIO::ReadFile(".\\src\\engine\\gl\\shaders\\basicFrag.frag");

    _basicShader = std::make_unique<Shader>("Basic");
    _basicShader->Load(vertexShaderSource, fragmentShaderSource);

}
