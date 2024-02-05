#include <engine/message/EventQueue.h>

void EventQueue::PostEvent(const Event& event) {
    _queue.push(event);
}

bool EventQueue::PollEvent(Event& event) {
    if (!_queue.empty()) {
        event = _queue.front();
        _queue.pop();
        return true;
    }
    return false;
}

EventQueue eventQueue;
