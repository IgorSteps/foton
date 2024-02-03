#pragma once
#include <engine/graphics/Sprite.h>

class SphereSprite : public Sprite
{
public:
	SphereSprite(const std::string& name, float radius, int segments, int stacks);
	virtual void Init() override;
	virtual void Update(float dt) override;
	virtual void Draw() override;
private:
	float _radius;
	int _segments, _stacks;
	void generateSphere(std::vector<float>& vertices, std::vector<unsigned int>& indices) const;
};