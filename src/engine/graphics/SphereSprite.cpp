#include <engine/graphics/SphereSprite.h>
#include <numbers>
#include "glm/gtc/matrix_transform.hpp"
#include "glm/ext.hpp"


SphereSprite::SphereSprite(const std::string& name, float radius, int segments, int stacks)
	: Sprite(name, 0, 0),
	_radius(radius),
	_segments(segments),
	_stacks(stacks)
{
    position = glm::vec3(1.0f, 0.0, 1.0f);
}

float SphereSprite::GetRadius() const
{
    return _radius;
}

void SphereSprite::Init()
{
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

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

void SphereSprite::Update(float dt)
{
}

void SphereSprite::Draw(std::unique_ptr<Shader>& shader)
{
    // colour
    auto colorLocation = shader->GetUniformLocation("myColor");
    glUniform4f(colorLocation, 1.0f, 0.5f, 0.0f, 1.0f);

    // model
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    auto modelLocation = shader->GetUniformLocation("u_model");
    glUniformMatrix4fv(modelLocation, 1, GL_FALSE, glm::value_ptr(model));

    _buffer->Bind(false);
    _buffer->Draw();
}

