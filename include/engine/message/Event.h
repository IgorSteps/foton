#pragma once

enum class EventType {
    None = 0,
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    LookAround,
    Zoom
};

class Event {
public:
    Event() : type(EventType::None), xoffset(0.0f), yoffset(0.0f) {}
    Event(EventType type, float xOff = 0.0f, float yOff = 0.0f)
        : type(type), xoffset(xOff), yoffset(yOff) {}

    EventType type;
    float xoffset; // For mouse movement and zoom
    float yoffset; // For mouse movement and zoom
};