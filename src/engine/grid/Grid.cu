#include <engine/grid/Grid.cuh>

__host__ Grid::Grid(std::vector<Sphere>& spheres)
{
    _numSpheres = spheres.size();
    _h_Spheres = spheres;
    _d_Spheres = _h_Spheres;
    ComputeSceneBoundingBox();
    ComputeGridResolution();
    // Populate cells:
    _cellSize = _gridSize / _gridResolution;
    int totalCells = _gridResolution.x * _gridResolution.y * _gridResolution.z;
    _h_Cells.resize(totalCells);
    

    Populate();
    // Copy to the GPU:
    CopyCellsToDevice();
    std::cout << "Finish setting up Grid" << std::endl;
}

__host__ Grid::~Grid() 
{

}

__device__ bool Grid::Intersect(const Ray& ray, float tMin, float tMax, HitData& hit)
{
    glm::vec3 normalisedRayDir = glm::normalize(ray.direction);
    // Check if the ray intersects the grid.
    float t0 = tMin, t1 = tMax;
    for (int i = 0; i < 3; ++i) 
    {
        float invDir = 1.0f / normalisedRayDir[i];
        float tNear = (_gridMin[i] - ray.origin[i]) * invDir;
        float tFar = (_gridMax[i] - ray.origin[i]) * invDir;
        if (tNear > tFar) 
        {
            float temp = tNear;
            tNear = tFar;
            tFar = temp;
        }
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1) {
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
            deltaT[i] = -_cellSize[i] / normalisedRayDir[i];
            step[i] = -1;
        }
    }
    
    // Traverse.
    while (true) {
        int cellIdx = GetCellIndex(cell.x, cell.y, cell.z);
        if (_d_Cells[cellIdx].Intersect(
            thrust::raw_pointer_cast(_d_Spheres.data()),
            _d_Cells[cellIdx].GetNumSpheres(),
            ray, tMin, tMax, hit)) 
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

__host__ void Grid::ComputeSceneBoundingBox()
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

__host__ void Grid::ComputeGridResolution()
{
    int numOfSpheres = _h_Spheres.size();
    float volume = _gridSize.x * _gridSize.y * _gridSize.z;
    float cubeRoot = std::pow(lambda * numOfSpheres / volume, 1.0f / 3.0f);

    _gridResolution = glm::vec3(_gridSize * cubeRoot);
}

__host__ void Grid::Populate()
{
    for (int i = 0; i < _h_Spheres.size(); ++i)
    {
        const Sphere& sphere = _h_Spheres[i];
        glm::vec3 sphereBBoxMin = sphere.GetCenter() - sphere.GetRadius();
        glm::vec3 sphereBBoxMax = sphere.GetCenter() + sphere.GetRadius();

        glm::ivec3 minCell = glm::floor(sphereBBoxMin / _cellSize);
        glm::ivec3 maxCell = glm::ceil(sphereBBoxMax / _cellSize);

        minCell = glm::clamp(minCell, glm::ivec3(0), glm::ivec3(_gridResolution - 1.0f));
        maxCell = glm::clamp(maxCell, glm::ivec3(0), glm::ivec3(_gridResolution - 1.0f));

        for (int z = minCell.z; z <= maxCell.z; ++z)
        {
            for (int y = minCell.y; y <= maxCell.y; ++y)
            {
                for (int x = minCell.x; x <= maxCell.x; ++x)
                {
                   
                    int cellIdx = GetCellIndex(x,y,z);
                    _h_Cells[cellIdx].Add(i);
                    
                }
            }
        }
    }
}

__host__ void Grid::CopyCellsToDevice()
{
    // Allocate memory for cells on the device.
    size_t numCells = _h_Cells.size();
    CUDA_CHECK_ERROR(cudaMalloc((void**)&_d_Cells, numCells * sizeof(Cell)));

    // Allocate and copy each cell from host to device.
    for (int i = 0; i < numCells; ++i)
    {
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