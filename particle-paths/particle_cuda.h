#ifndef PARTICLE_CUDA_H
#define PARTICLE_CUDA_H

#ifdef USE_CUDA

#include "particle.h"

void cuda_initPhysics(int maxParticles);
void cuda_cleanupPhysics();
void cuda_updatePhysics(SimulationState &simState);

#endif
#endif
