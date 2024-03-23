#pragma once

enum class EventType {
    None = 0,
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    LookAround,
    WindowResize
};

class Event {
public:
    Event() 
        : type(EventType::None), xoffset(0.0f), yoffset(0.0f), width(0), height(0) {}
    Event(EventType type, float xOff = 0.0f, float yOff = 0.0f, int w = 0.0f, int h = 0.0f)
        : type(type), xoffset(xOff), yoffset(yOff), width(w), height(h) {}

    EventType type;

    // For mouse movement:
    float xoffset;
    float yoffset;

    // For window resize:
    int width;
    int height;
};