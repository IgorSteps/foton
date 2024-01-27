#include <renderer/Engine.h>

int main()
{
    try {
        Engine engine;
        engine.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}


//
//int main()
//{
//    auto start = high_resolution_clock::now();
//
//    // Image
//    Image image = Image();
//    Camera camera = Camera(image.Width(), image.Height(), 1.0f, 2.0f, glm::vec3(0, 0, 0));
//
//    // Sphere
//    Sphere sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f);
//
//    // Render
//    Renderer r(image, camera, sphere);
//    r.Render();
//
//    // After function call
//    auto stop = high_resolution_clock::now();
//    auto duration = duration_cast<microseconds>(stop - start);
//    // To get the value of duration use the count()
//    // member function on the duration object
//    std::clog << "Time taken: " << duration.count() << " microseconds" << std::flush;
//}