#version 450 core
layout(location = 0) in vec3 aPos;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

out vec4 vertexColor;

void main() {
    gl_Position = u_projection * u_view * u_model * vec4(aPos, 1.0);
    vertexColor = vec4(0.5, 0.0, 0.0, 1.0);
}