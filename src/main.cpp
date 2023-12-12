#include <renderer/renderer.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;


int main()
{

    // Image
    Image image = Image();
    Camera camera = Camera(image.Width(), image.Height(), 1.0f, 2.0f, glm::vec3(0, 0, 0));

    // Sphere
    Sphere sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f);

    int numIterations = 10; // Set the number of iterations
    long long totalAverageDuration = 0; // Variable to accumulate average durations

    for (int i = 0; i < numIterations; i++) {
        Renderer r(image, camera, sphere);
        long long averageDuration = r.Render();
        totalAverageDuration += averageDuration;
    }

    long long overallAverage = totalAverageDuration / numIterations;
    std::clog << "Overall average time taken by WriteColour: " << overallAverage << " microseconds" << std::flush;

    return 0;
}