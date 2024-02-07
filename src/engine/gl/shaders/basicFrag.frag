#version 450 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D u_texture;

void main() {
    FragColor = texture(u_texture, TexCoords);
}