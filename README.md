# Foton

Foton is a real time ray-tracer build in C++ with CUDA. It has:

- Lights and shadows using Phong Illumination.
- Acceleration using a Grid data structure.
- Moving camera that can be controlled using WASD keys and a mouse.

[Jira Board](https://stepanen.atlassian.net/jira/software/projects/FTN/boards/2)

## Choosing demo scene
In the `Engine.cpp`, in the `init()` choose the desired function to populate the world and in the `update()` choose a kernel to run
(Simple, Grid, Phong). You can also uncomment lines of code that allow the light to move around a scene.

## Running locally

### Prerequisites 
- Visual Studio
- NVIDIA GPU
- OpenGL v4.5
- CUDA v12.3.52

To run, git clone and open Visual Studio solution and click Run.
