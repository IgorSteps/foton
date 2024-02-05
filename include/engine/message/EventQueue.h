#pragma once
#include <queue>
#include <engine/message/Event.h>
class EventQueue
{
public:
	void PostEvent(const Event& event);
	bool PollEvent(Event& event);
private:
	std::queue<Event> _queue;
};