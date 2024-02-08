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
    generateSphere(vertices, indices);

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

 bool SphereSprite::Intersects(const Ray& r)
{
    glm::vec3 oc = r.Origin() - position;

    float a = glm::dot(r.Direction(), r.Direction());
    float b = 2.0f * glm::dot(oc, r.Direction());
    float c = glm::dot(oc, oc) - pow(_radius, 2);

    float discriminant = pow(b, 2) - 4 * a * c;

    return discriminant >= 0;
}

/// <summary>
/// Generate sphere vertices and indices using UV sphere method.
/// Reference: https://www.songho.ca/opengl/gl_sphere.html
/// </summary>
void SphereSprite::generateSphere(std::vector<float>& vertices, std::vector<unsigned int>& indices) const
{
    // Generate vertices.
    for (int stack = 0; stack <= _stacks; ++stack) 
    {
        float phi = std::numbers::pi * float(stack) / float(_stacks); // Latitude
        for (int segment = 0; segment <= _segments; ++segment) 
        {
            float theta = 2.0 * std::numbers::pi * float(segment) / float(_segments); // Longitude

            float x = _radius * sin(phi) * cos(theta);
            float y = _radius * sin(phi) * sin(theta);
            float z = _radius * cos(phi);

            // Add (x, y, z) to your vertices list
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }

    // Generate indices
    for (int stack = 0; stack < _stacks; ++stack) 
    {
        for (int segment = 0; segment < _segments; ++segment) 
        {
            int first = (stack * (_segments + 1)) + segment;
            int second = first + _segments + 1;

            // First triangle
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            // Second triangle
            indices.push_back(first + 1);
            indices.push_back(second);
            indices.push_back(second + 1);
        }
    }
}

