#ifndef GPU_RENDERER_H
#define GPU_RENDERER_H

#include <GL/glew.h>
#include <vector>
#include "particle.h"

extern bool GPUinstance;
bool initGPU();

void drawGPU(const std::vector<Particle> &particles,
             int robotIdx,
             int goalIdx);

void cleanupGPU();

#endif
