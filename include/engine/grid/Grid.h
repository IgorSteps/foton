#pragma once
#include "Cell.h"
#include "engine/hittables/Sphere.h"
struct Grid {
    int gridSize; // Number of cells along one dimension
    float cellSize; // Size of each cell
    glm::vec3 gridMin; // Minimum corner of the grid

    std::vector<Cell> cells; // 1D array to store cells in a 3D manner
    std::vector<Sphere> spheres; // List of all spheres

    Grid(int gridSize, float cellSize, float gridMin)
        : gridSize(gridSize), cellSize(cellSize), gridMin(gridMin) 
    {
        cells.resize(gridSize * gridSize * gridSize);
    }

    // Add a sphere to the grid
    void AddSphere(const Sphere& sphere) 
    {
    
    }

    // Map a sphere to its corresponding cells
    void MapSphereToCells(const Sphere& sphere, int sphereIndex) {
        // Calculate the bounding box of the sphere
        float3 minCorner = sphere.GetCenter() - make_float3(sphere.radius);
        float3 maxCorner = sphere.center + make_float3(sphere.radius);

        // Convert bounding box to cell coordinates
        int3 minCell = getCellCoords(minCorner);
        int3 maxCell = getCellCoords(maxCorner);

        // Loop through the affected cells
        for (int z = minCell.z; z <= maxCell.z; ++z) {
            for (int y = minCell.y; y <= maxCell.y; ++y) {
                for (int x = minCell.x; x <= maxCell.x; ++x) {
                    int cellIndex = getCellIndex(x, y, z);
                    cells[cellIndex].addSphere(sphereIndex);
                }
            }
        }
    }

    // Convert world coordinates to cell coordinates
    glm::vec3 GetCellCoords(const glm::vec3& pos) const {
        float x = static_cast<int>((pos.x - gridMin.x) / cellSize);
        float y = static_cast<int>((pos.y - gridMin.y) / cellSize);
        float z = static_cast<int>((pos.z - gridMin.z) / cellSize);
        return glm::vec3(x, y, z);
    }
};

class Grid {
public:
    Grid(const std::vector<Sphere>& spheres)
    {
        _spheres = spheres;
        ComputeSceneBoundingBox();
        ComputeGridResolution();
        _cellSize = _gridSize / _gridResolution;
        _cells.resize(_gridResolution.x * _gridResolution.y * _gridResolution.z);
        Populate();
    }    

private:
    glm::vec3 _gridResolution;
    glm::vec3 _gridSize;
    glm::vec3 _cellSize;
    glm::vec3 _gridMin;
    glm::vec3 _gridMax; 
    const int lambda = 5;
    // 1D array that stores 3D coords in the following format:
    // First all x coords, then all y coords and lastly all z coords.
    std::vector<Cell> _cells;
    std::vector<Sphere> _spheres;

    void ComputeSceneBoundingBox() 
    {
        _gridMin = _spheres[0].GetCenter() - _spheres[0].GetRadius();
        _gridMax = _spheres[0].GetCenter() + _spheres[0].GetRadius();
        for (const Sphere& sphere : _spheres) 
        {
            glm::vec3 sphereMin = sphere.GetCenter() - sphere.GetRadius();
            glm::vec3 sphereMax = sphere.GetCenter() + sphere.GetRadius();
            _gridMin = glm::min(_gridMin, sphereMin);
            _gridMax = glm::min(_gridMax, sphereMax);
        }
        _gridSize = _gridMax - _gridMin;
    }

    void ComputeGridResolution()
    {
        int numOfSpheres = _spheres.size();
        float volume = _gridSize.x * _gridSize.y * _gridSize.z;
        float cubeRoot = std::pow(lambda * numOfSpheres / volume, 1 / 3);

        _gridResolution = glm::vec3(_gridSize * cubeRoot);
    }

    void Populate() 
    {
        for (int i = 0; i < _spheres.size(); ++i) 
        {
            const Sphere& sphere = _spheres[i];
            glm::vec3 sphereBBoxMin = sphere.GetCenter() - sphere.GetRadius();
            glm::vec3 sphereBBoxMax = sphere.GetCenter() + sphere.GetRadius();

            glm::vec3 minCell = sphereBBoxMin / _cellSize;
            glm::vec3 maxCell = sphereBBoxMax / _cellSize;

            for (int z = minCell.z; z <= maxCell.z; ++z) 
            {
                for (int y = minCell.y; y <= maxCell.y; ++y) 
                {
                    for (int x = minCell.x; x <= maxCell.x; ++x)
                    {
                        int cellIdx = GetCellIndex(x, y, z);
                        _cells[cellIdx].Add(i);
                    }
                }
            }
        }
    }

    glm::vec3 GetCellCoords(const glm::vec3& worldPos) const 
    {
        glm::vec3 coords = (worldPos - _gridMin) / _cellSize;
        return glm::clamp(coords, glm::vec3(0.0f), _gridResolution - 1.0f);

    }

    int GetCellIndex(int x, int y, int z) const 
    {
        // Convert 3D coords to 1D index using x + y * width + z * width * height.
        return x + y * _gridResolution.x + z * _gridResolution.x * _gridResolution.y;
    }
};