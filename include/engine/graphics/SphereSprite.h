#pragma once
#include <engine/graphics/Sprite.h>
#include <engine/gl/Shader.h>
#include <engine/Ray.h>

class SphereSprite : public Sprite
{
public:
	SphereSprite(const std::string& name, float radius, int segments, int stacks);
	float GetRadius() const;
	virtual void Init() override;
	virtual void Update(float dt) override;
	virtual void Draw(std::unique_ptr<Shader>& shader) override;
	bool Intersects(const Ray& ray);
private:
	float _radius;
	int _segments, _stacks;
	void generateSphere(std::vector<float>& vertices, std::vector<unsigned int>& indices) const;
};