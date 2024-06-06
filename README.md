# Foton

A Real-Time raytracer build in C++ with CUDA.

[Jira Board](https://stepanen.atlassian.net/jira/software/projects/FTN/boards/2)

## Choosing demo scene
In the Engine.cpp file, in the `init()` method uncomment desired function to populate the world and in the `update()` function choose 
the the kernel(Simple, Grid, Phong).

## Running locally

### Prerequisites 
- Visual Studio
- NVIDIA GPU
- OpenGL v4.5
- CUDA v12.3.52

To run, git clone and open Visual Studio solution and click Run.
