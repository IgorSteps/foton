#include <engine/grid/Grid.cuh>

__host__ Grid::Grid(std::vector<Sphere>& spheres)
{
    _numSpheres = spheres.size();
    _h_Spheres = spheres;
    _d_Spheres = _h_Spheres;
    ComputeGridSize();
    ComputeGridResolution();
    Populate();
    CopyCellsToDevice();
}

// Intersect traverses the grid and checks for sphere hits using 3D-DDA algorithm.
__device__ bool Grid::Intersect(const Ray& ray, HitData& hit)
{
    glm::vec3 normalisedRayDir = glm::normalize(ray.direction);

    // Check if the ray intersects the grid using AABB test.
    float tGridEntry = 0.00001f, tGridExit = INFINITY;
    for (int i = 0; i < 3; ++i) 
    {
        float tMin = (_gridMin[i] - ray.origin[i]) / normalisedRayDir[i];
        float tMax = (_gridMax[i] - ray.origin[i]) / normalisedRayDir[i];
        // Make sure tMin is always smaller than tMax.
        if (tMin > tMax) 
        {
            float tempTMin = tMin;
            tMin = tMax;
            tMax = tempTMin;
        }

        // Also set a valid hit interval for the ray to the grid boundaries.
        tGridEntry = glm::max(tMin, tGridEntry);
        tGridExit = glm::min(tMax, tGridExit);
        if (tGridEntry > tGridExit) 
        { 
            return false;
        }
    }

    // Convert ray's origin to the cell coordinates.
    glm::vec3 gridRelativeRayOrigin = ray.origin - _gridMin;
    glm::ivec3 cell = GetCellCoords(ray.origin);

    // Calculate initial t, deltaT and step values.
    glm::vec3 deltaT = glm::vec3(0.0f), t = glm::vec3(0.0f);
    glm::ivec3 step = glm::ivec3(0);
    for (int i = 0; i < 3; ++i) 
    { 
        if (normalisedRayDir[i] > 0) // Positive ray direction
        { 
            t[i] = ((cell[i] + 1) * _cellSize[i] - gridRelativeRayOrigin[i]) / normalisedRayDir[i];
            deltaT[i] = _cellSize[i] / normalisedRayDir[i];
            step[i] = 1;
        }
        else
        {
            t[i] = (cell[i] * _cellSize[i] - gridRelativeRayOrigin[i]) / normalisedRayDir[i];
            deltaT[i] = _cellSize[i] / normalisedRayDir[i];
            step[i] = -1;
        }
    }

    // Traverse.
    while (true) {
        int cellIdx = GetCellIndex(cell.x, cell.y, cell.z);
        const Cell& currentCell = _d_Cells[cellIdx];
        Sphere* spheres = thrust::raw_pointer_cast(_d_Spheres.data());
        int numSpheres = currentCell.GetNumSpheres();
        if (currentCell.Intersect(spheres, numSpheres, ray, tGridEntry, tGridExit, hit)) // Using grid boundaries for hit interval.
        {
            return true;
        }

        // Determine the next cell to step to.
        if (t.x < t.y && t.x < t.z) {
            t.x += deltaT.x;
            cell.x += step.x;
        }
        else if (t.y < t.z) {
            t.y += deltaT.y;
            cell.y += step.y;
        }
        else {
            t.z += deltaT.z;
            cell.z += step.z;
        }

        // Out-of-bounds check.
        if (
            cell.x < 0 || cell.x >= _gridResolution.x ||
            cell.y < 0 || cell.y >= _gridResolution.y ||
            cell.z < 0 || cell.z >= _gridResolution.z
        ) 
        {
            break;
        }
    }

    return false;
}

// ComputeGridSize calcualtes the sum of bboxes of all spheres which is the grid size.
__host__ void Grid::ComputeGridSize()
{
    _gridMin = _h_Spheres[0].GetCenter() - _h_Spheres[0].GetRadius();
    _gridMax = _h_Spheres[0].GetCenter() + _h_Spheres[0].GetRadius();
    for (const Sphere& sphere : _h_Spheres)
    {
        glm::vec3 sphereMin = sphere.GetCenter() - sphere.GetRadius();
        glm::vec3 sphereMax = sphere.GetCenter() + sphere.GetRadius();
        _gridMin = glm::min(_gridMin, sphereMin);
        _gridMax = glm::max(_gridMax, sphereMax);
    }
    _gridSize = _gridMax - _gridMin;
}

// ComputeGridResolution using fancy maths.
__host__ void Grid::ComputeGridResolution()
{
    int numOfSpheres = _h_Spheres.size();
    float volume = _gridSize.x * _gridSize.y * _gridSize.z;
    float cubeRoot = std::pow(lambda * numOfSpheres / volume, 1.0f / 3.0f);

    _gridResolution = glm::vec3(_gridSize * cubeRoot);
}

// Populate populates the grid cells with sphere indexes.
__host__ void Grid::Populate()
{
    _cellSize = _gridSize / _gridResolution;
    _h_Cells.resize(_gridResolution.x * _gridResolution.y * _gridResolution.z);

    for (int sphereIdx = 0; sphereIdx < _h_Spheres.size(); ++sphereIdx)
    {
        const Sphere& sphere = _h_Spheres[sphereIdx];
        glm::vec3 sphereBBoxMin = sphere.GetCenter() - sphere.GetRadius();
        glm::vec3 sphereBBoxMax = sphere.GetCenter() + sphere.GetRadius();

        // Floor and ceil to get higher and lower cell indexes.
        glm::vec3 minCell = glm::floor(sphereBBoxMin / _cellSize);
        glm::vec3 maxCell = glm::ceil(sphereBBoxMax / _cellSize);

        // Clamp to make sure we are within the grid's boundaries.
        minCell = glm::clamp(minCell, glm::vec3(0.0f), glm::vec3(_gridResolution - 1.0f));
        maxCell = glm::clamp(maxCell, glm::vec3(0.0f), glm::vec3(_gridResolution - 1.0f));

        // Insert sphere indexes.
        for (int z = minCell.z; z <= maxCell.z; ++z)
        {
            for (int y = minCell.y; y <= maxCell.y; ++y)
            {
                for (int x = minCell.x; x <= maxCell.x; ++x)
                {
                    int cellIdx = GetCellIndex(x,y,z);
                    _h_Cells[cellIdx].Add(sphereIdx);
                }
            }
        }
    }
}

// CopyCellsToDevice allocates and copies cells array and internal cell data to the device.
__host__ void Grid::CopyCellsToDevice()
{
    size_t numCells = _h_Cells.size();
    CUDA_CHECK_ERROR(cudaMalloc(&_d_Cells, numCells * sizeof(Cell)));
    for (int i = 0; i < numCells; ++i)
    {
        // Internal cell data must be copied to the device as well.
        _h_Cells[i].AllocateDeviceMemory();
        _h_Cells[i].CopyToDevice();
        CUDA_CHECK_ERROR(cudaMemcpy(&_d_Cells[i], &_h_Cells[i], sizeof(Cell), cudaMemcpyHostToDevice));
    }
}

// GetCellCoords gets cell coordinates relative to the worldPos.
__device__ glm::vec3 Grid::GetCellCoords(const glm::vec3& worldPos) const
{
    glm::vec3 gridRelativeCoords = (worldPos - _gridMin) / _cellSize;
    // Clamp to make sure the cell is within grid's boundaries.
    return glm::clamp(glm::floor(gridRelativeCoords), glm::vec3(0.0f), _gridResolution - 1.0f);
}

// GetCellIndex convert's 3D coordinates to 1D index.
__device__ __host__ int Grid::GetCellIndex(int x, int y, int z) const
{
    return x + y * _gridResolution.x + z * _gridResolution.x * _gridResolution.y;
}