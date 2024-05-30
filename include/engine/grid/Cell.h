#pragma once
#include <vector>

class Cell {
public:
    void Add(int sphereID) 
    {
        _sphereIds.push_back(sphereID);
    }

private:
    std::vector<int> _sphereIds;
};