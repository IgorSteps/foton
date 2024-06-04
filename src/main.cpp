#include <engine/Engine.h>

int main(int argc, char* argv[])
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