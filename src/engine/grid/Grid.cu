#include <engine/grid/Grid.cuh>

#define CUDA_CHECK_ERROR(call)                                          \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error " << err << " at " << __FILE__ <<  \
            ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

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
    glm::vec3 rayGridOrigin = ray.origin - _gridMin;
    glm::ivec3 floredVal = glm::floor(rayGridOrigin / _cellSize);
    glm::ivec3 originCell = glm::clamp(floredVal, glm::ivec3(0.0f), glm::ivec3(_gridResolution - 1.0f));
    glm::vec3 normalisedRayDir = glm::normalize(ray.direction);
    glm::vec3 deltaT, t;
    glm::ivec3 step;

    // AABB (Axis-Aligned Bounding Box) intersection test
    float t0 = tMin, t1 = tMax;
    for (int i = 0; i < 3; ++i) {
        float invDir = 1.0f / normalisedRayDir[i];
        float tNear = (_gridMin[i] - ray.origin[i]) * invDir;
        float tFar = (_gridMax[i] - ray.origin[i]) * invDir;
        if (tNear > tFar) {
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

    // Compute initial t and deltaT for each axis
    for (int i = 0; i < 3; ++i) {
        if (normalisedRayDir[i] > 0) {
            t[i] = ((originCell[i] + 1) * _cellSize[i] - rayGridOrigin[i]) / normalisedRayDir[i];
            deltaT[i] = _cellSize[i] / std::abs(normalisedRayDir[i]);
            step[i] = 1;
        }
        else {
            t[i] = (originCell[i] * _cellSize[i] - rayGridOrigin[i]) / normalisedRayDir[i];
            deltaT[i] = -_cellSize[i] / std::abs(normalisedRayDir[i]);
            step[i] = -1;
        }
    }

    float currentT = t0;
    while (currentT <= t1) {
        // Calculate cell index correctly
        int cellIdx = GetCellIndex(originCell.x, originCell.y, originCell.z);
        if (_d_Cells[cellIdx].Intersect(thrust::raw_pointer_cast(_d_Spheres.data()), _d_Cells[cellIdx]._h_NumSpheres, ray, tMin, tMax, hit)) {
            tMax = hit.t;
            return true;
        }

        // Determine the next cell to step to:
        if (t.x < t.y && t.x < t.z) {
            currentT = t.x;
            t.x += deltaT.x;
            originCell.x += step.x;
        }
        else if (t.y < t.z) {
            currentT = t.y;
            t.y += deltaT.y;
            originCell.y += step.y;
        }
        else {
            currentT = t.z;
            t.z += deltaT.z;
            originCell.z += step.z;
        }

        // Check if we are out of bounds
        if (originCell.x < 0 || originCell.x >= _gridResolution.x ||
            originCell.y < 0 || originCell.y >= _gridResolution.y ||
            originCell.z < 0 || originCell.z >= _gridResolution.z) {
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
        _gridMax = glm::min(_gridMax, sphereMax);
    }
    _gridSize = _gridMax - _gridMin;
}

__host__ void Grid::ComputeGridResolution()
{
    int numOfSpheres = _h_Spheres.size();
    float volume = _gridSize.x * _gridSize.y * _gridSize.z;
    float cubeRoot = std::pow(lambda * numOfSpheres / volume, 1 / 3);

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
                   
                    int cellIdx = z * _gridResolution.x * _gridResolution.y + y * _gridResolution.x + x; ;
                    std::cout << "Adding sphereIdx: " << i << " to Cell at index: " << cellIdx << std::endl;
                    _h_Cells[cellIdx].Add(i);
                    
                }
            }
        }
    }
}

__host__ void Grid::CopyCellsToDevice()
{
    // Allocate memory for cells on the device
    size_t numCells = _h_Cells.size();
    CUDA_CHECK_ERROR(cudaMalloc((void**)&_d_Cells, numCells * sizeof(Cell)));

    // Allocate memory for each cell's internal device data
    for (int i = 0; i < numCells; ++i)
    {
        std::cout << "Allocating device memory for cell " << i << std::endl;
        _h_Cells[i].AllocateDeviceMemory();
    }

    // Copy each cell from host to device
    for (int i = 0; i < numCells; ++i)
    {
        std::cout << "Copying cell " << i << " with " << _h_Cells[i]._h_NumSpheres << " spheres to device." << std::endl;
        _h_Cells[i].CopyToDevice();
        CUDA_CHECK_ERROR(cudaMemcpy(&_d_Cells[i], &_h_Cells[i], sizeof(Cell), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

__host__ glm::vec3 Grid::GetCellCoords(const glm::vec3& worldPos) const
{
    glm::vec3 coords = (worldPos - _gridMin) / _cellSize;
    return glm::clamp(coords, glm::vec3(0.0f), _gridResolution - 1.0f);

}

__device__ __host__ int Grid::GetCellIndex(int x, int y, int z) const
{
    // Convert 3D coords to 1D index using x + y * width + z * width * height.
    return x + y * _gridResolution.x + z * _gridResolution.x * _gridResolution.y;
}