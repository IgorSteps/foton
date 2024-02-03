#pragma once
#include <engine/graphics/Sprite.h>

class QuadSprite : public Sprite 
{
public:
    QuadSprite(const std::string& name, float width, float height);

    virtual void Init() override;
    virtual void Update(float dt) override;
    virtual void Draw() override;
};