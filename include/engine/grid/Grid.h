#pragma once
#include "Cell.h"
#include "engine/hittables/Sphere.h"
#include "engine/Ray.h"

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

    bool Intersect(const Ray& ray, float tMin, float tMax, HitData& hit)
    {
        glm::vec3 rayGridOrigin = ray.origin - _gridMin;
        glm::vec3 originCell = rayGridOrigin / _cellSize;
        glm::vec3 normalisedRayDir = glm::normalize(ray.direction);
        glm::vec3 deltaT = glm::vec3(0), t = glm::vec3(0);

        // TODO: Move elswhere
        // AABB (Axis-Aligned Bounding Box) intersection test
        float t0 = tMin, t1 = tMax;
        for (int i = 0; i < 3; ++i)
        {
            float invDir = 1.0f / normalisedRayDir[i];
            float tNear = (_gridMin[i] - ray.origin[i]) * invDir;
            float tFar = (_gridMax[i] - ray.origin[i]) * invDir;

            if (tNear > tFar) std::swap(tNear, tFar);

            t0 = tNear > t0 ? tNear : t0;
            t1 = tFar < t1 ? tFar : t1;

            if (t0 > t1) return; // Ray does not intersect the grid
        }

        // Initilise t0 and deltaT;
        for (int i = 0; i < 3; ++i)
        {
            if (normalisedRayDir[i] > 0) // Positive direction.
            {
                t[i] = ((floor(originCell[i]) + 1) * _cellSize[i] - rayGridOrigin[i]) / normalisedRayDir[i];
            }
            else // Negative direction.
            {
                t[i] = ((ceil(originCell[i]) - 1) * _cellSize[i] - rayGridOrigin[i]) / normalisedRayDir[i];
            }
            deltaT[i] = _cellSize[i] / std::abs(normalisedRayDir[i]);
        }

        float currentT = 0.0f;
        while (true)
        {
            // Check if the ray intersects any spheres in the current cell
            int cellIdx = GetCellIndex(static_cast<int>(originCell.x), static_cast<int>(originCell.y), static_cast<int>(originCell.z));
            const Cell& cell = _cells[cellIdx];

            // Check intersections with spheres in the current cell
            if (cell.Intersect(_spheres.data(), ray, tMin, tMax, hit))
            {
                tMax = hit.t;
                return true;
            }

            // Determine the next cell to step to:
            if (t.x < t.y)
            {
                currentT = t.x;
                t.x += deltaT.x;
                if (normalisedRayDir.x > 0) // Positive direction.
                {
                    originCell.x += 1;
                }
                else // Negatve direction.
                {
                    originCell.x -= 1;
                }
            }
            else if (t.y < t.z)
            {
                currentT = t.y;
                t.y += deltaT.y;
                if (normalisedRayDir.y > 0) // Positive direction.
                {
                    originCell.y += 1;
                }
                else // Negatve direction.
                {
                    originCell.y -= 1;
                }
            }
            else
            {
                currentT = t.z;
                t.z += deltaT.z;
                if (normalisedRayDir.z > 0) // Positive direction.
                {
                    originCell.z += 1;
                }
                else // Negatve direction.
                {
                    originCell.z -= 1;
                }
            }

            if (originCell.x < 0 || originCell.x > _gridMax.x - 1.0f ||
                originCell.y < 0 || originCell.y > _gridMax.y - 1.0f ||
                originCell.z < 0 || originCell.z > _gridMax.z - 1.0f)
            {
                break;
            }
        }
        
        return false;
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

    __device__ __host__ int GetCellIndex(int x, int y, int z) const 
    {
        // Convert 3D coords to 1D index using x + y * width + z * width * height.
        return x + y * _gridResolution.x + z * _gridResolution.x * _gridResolution.y;
    }
};