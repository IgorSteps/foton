#include "engine/graphics/QuadSprite.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/ext.hpp"
QuadSprite::QuadSprite(const std::string& name, float width, float height) 
    : Sprite(name, width, height)
{
    position = glm::vec3(-1.0f, 0.0f, 1.0f);
}

void QuadSprite::Init()
{
    // top left corner at 0,0.
    std::vector<float>  vertices = {
        // positions        
        0.5f,  0.5f, 0.0f,  
        0.5f, -0.5f, 0.0f,  
        -0.5f, -0.5f, 0.0f,  
        -0.5f,  0.5f, 0.0f,   
    };

    std::vector<unsigned int> indices = {
        0, 1, 3,
        1, 2, 3
    };

    // Create a GLBuffer object
    _buffer = std::make_unique<GLBuffer>(GL_FLOAT, GL_TRIANGLES);

    // Define attribute information
    AttributeInfo positionAttrib(0, 3, 0);
    _buffer->AddAttributeLocation(positionAttrib);

    // Set vertex and element data
    _buffer->SetVertexData(vertices);
    _buffer->SetElementData(indices);

    // Upload data to the GPU
    _buffer->UploadData();
    _buffer->Unbind();
}

void QuadSprite::Update(float dt)
{
}

void QuadSprite::Draw(std::unique_ptr<Shader>& shader)
{
    // colour
    auto colorLocation = shader->GetUniformLocation("myColor");
    glUniform4f(colorLocation, 0.0f, 1.0f, 0.0f, 1.0f);

    // model
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    auto modelLocation = shader->GetUniformLocation("u_model");
    glUniformMatrix4fv(modelLocation, 1, GL_FALSE, glm::value_ptr(model));

    _buffer->Bind(false);
    _buffer->Draw();
}
