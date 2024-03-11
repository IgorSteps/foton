#pragma once
#include <memory>
#include <engine/gl/Shader.h>
#include <engine/gl/glBuffer.h>
#include <engine/gl/PBOBuffer.h>
#include <glm/glm.hpp>

class Sprite
{
public:
    Sprite();

    void Init();
    void Update(float dt);
    void Draw(std::unique_ptr<Shader>& shader);

private:
    float _width, _height;
    std::string _name;
    glm::vec3 position;
    std::unique_ptr<GLBuffer> _buffer;
};