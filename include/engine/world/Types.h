#pragma once
#include <bitset>


/*
	Credit: https://austinmorlan.com/posts/entity_component_system/
*/

// A simple type alias
using Entity = int;

// Used to define the size of arrays later on
const Entity MAX_ENTITIES = 5000;

// A simple type alias
using ComponentType = int;

// Used to define the size of arrays later on
const ComponentType MAX_COMPONENTS = 32;

// A simple type alias
using Signature = std::bitset<MAX_COMPONENTS>;