#include <utils/colour.h>

void WriteColour(std::ostream& out, glm::vec3 colour)
{
    // Write the translated [0,255] value of each colour component.
    out << static_cast<int>(255.999 * colour.x) << ' '
        << static_cast<int>(255.999 * colour.y) << ' '
        << static_cast<int>(255.999 * colour.z) << '\n';

}
