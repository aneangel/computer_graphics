#include "particle.h"
#include <cmath>
#include <cstdlib>

float boxsize = 5.0f;
int gridWidth = 35;

SimulationState simState;

const float spawnX = 0.0f;
const float spawnZ = 0.0f;
const int partPerFrame = 15;

void SimulationState::reset()
{
    particles.clear();
    grid.clear();
    particlesSpawned = 0;
    spawnY = boxsize / 2.0f - 0.5f;
    robotParticle = -1;
    goalParticle = -1;
    currentPath.clear();
    pathUpdateCounter = 0;
    goalReached = false;
    initialAgitationDone = false;
    settlingFrames = 0;
    elapsedTime = 0.0f;
    runCount++;

    replanLog.clear();

    obstacles.clear();
    {
        const float hs = boxsize / 2.0f;
        const float p = hs * 0.60f;
        const float m = hs * 0.30f;
        const float exS = hs * 0.10f;
        const float exD = hs * 0.088f;
        const float s = (boxsize / 5.0f);

        if (currentScenario == Scenario::STATIC_OBSTACLES)
        {

            // Bottom layer  y = -p
            obstacles.emplace_back(-p, -p, -p, exS, exS, exS);
            obstacles.emplace_back(0, -p, -p, exS, exS, exS);
            obstacles.emplace_back(p, -p, -p, exS, exS, exS);
            obstacles.emplace_back(-p, -p, 0, exS, exS, exS);
            obstacles.emplace_back(p, -p, 0, exS, exS, exS);
            obstacles.emplace_back(-p, -p, p, exS, exS, exS);
            obstacles.emplace_back(0, -p, p, exS, exS, exS);
            obstacles.emplace_back(p, -p, p, exS, exS, exS);

            // Middle layer  y = 0  (offset ±m to block diagonal shortcuts)
            obstacles.emplace_back(-m, 0, -m, exS, exS, exS);
            obstacles.emplace_back(m, 0, -m, exS, exS, exS);
            obstacles.emplace_back(-m, 0, m, exS, exS, exS);
            obstacles.emplace_back(m, 0, m, exS, exS, exS);

            // Top layer  y = +p  (rotated pattern vs bottom)
            obstacles.emplace_back(0, p, -p, exS, exS, exS);
            obstacles.emplace_back(-p, p, -m, exS, exS, exS);
            obstacles.emplace_back(p, p, -m, exS, exS, exS);
            obstacles.emplace_back(0, p, 0, exS, exS, exS);
            obstacles.emplace_back(-p, p, m, exS, exS, exS);
            obstacles.emplace_back(p, p, m, exS, exS, exS);
        }
        else if (currentScenario == Scenario::DYNAMIC_OBSTACLES)
        {
            // Bottom layer  y = -p  (lateral / upward velocities)
            obstacles.emplace_back(-p, -p, -p, exD, exD, exD, 0.5f * s, 0.4f * s, 0.3f * s);
            obstacles.emplace_back(0, -p, -p, exD, exD, exD, -0.4f * s, 0.6f * s, 0.5f * s);
            obstacles.emplace_back(p, -p, -p, exD, exD, exD, -0.6f * s, 0.3f * s, -0.4f * s);
            obstacles.emplace_back(-p, -p, 0, exD, exD, exD, 0.3f * s, 0.5f * s, -0.6f * s);
            obstacles.emplace_back(p, -p, 0, exD, exD, exD, -0.5f * s, 0.4f * s, 0.6f * s);
            obstacles.emplace_back(-p, -p, p, exD, exD, exD, 0.6f * s, 0.3f * s, -0.5f * s);
            obstacles.emplace_back(0, -p, p, exD, exD, exD, 0.4f * s, 0.6f * s, -0.3f * s);
            obstacles.emplace_back(p, -p, p, exD, exD, exD, -0.3f * s, 0.5f * s, 0.4f * s);

            // Middle layer  y = 0  (mostly horizontal velocities)
            obstacles.emplace_back(-m, 0, -m, exD, exD, exD, 0.6f * s, -0.3f * s, 0.5f * s);
            obstacles.emplace_back(m, 0, -m, exD, exD, exD, -0.5f * s, 0.4f * s, -0.6f * s);
            obstacles.emplace_back(-m, 0, m, exD, exD, exD, 0.4f * s, -0.6f * s, -0.4f * s);
            obstacles.emplace_back(m, 0, m, exD, exD, exD, -0.3f * s, 0.5f * s, 0.6f * s);

            // Top layer  y = +p  (lateral / downward velocities)
            obstacles.emplace_back(0, p, -p, exD, exD, exD, 0.5f * s, -0.4f * s, -0.3f * s);
            obstacles.emplace_back(-p, p, -m, exD, exD, exD, -0.4f * s, -0.6f * s, 0.5f * s);
            obstacles.emplace_back(p, p, -m, exD, exD, exD, 0.6f * s, -0.3f * s, -0.4f * s);
            obstacles.emplace_back(0, p, 0, exD, exD, exD, -0.3f * s, -0.5f * s, 0.6f * s);
            obstacles.emplace_back(-p, p, m, exD, exD, exD, 0.5f * s, -0.4f * s, -0.6f * s);
            obstacles.emplace_back(p, p, m, exD, exD, exD, -0.6f * s, -0.3f * s, 0.4f * s);
        }
    }
}

int gridHash(int x, int y, int z)
{
    return (x * 92837111) ^ (y * 689287499) ^ (z * 283923481);
}

int gridIndex(float pos)
{
    float halfSize = boxsize / 2.0f;
    return (int)((pos + halfSize) / cellSize);
}

void updateObstacles(SimulationState &simState)
{
    float halfSize = boxsize / 2.0f;

    // Move each obstacle and bounce off box walls
    for (auto &obs : simState.obstacles)
    {
        obs.cx += obs.vx * timeStep;
        obs.cy += obs.vy * timeStep;
        obs.cz += obs.vz * timeStep;

        if (obs.cx - obs.hx < -halfSize)
        {
            obs.cx = -halfSize + obs.hx;
            obs.vx = std::abs(obs.vx);
        }
        if (obs.cx + obs.hx > halfSize)
        {
            obs.cx = halfSize - obs.hx;
            obs.vx = -std::abs(obs.vx);
        }
        if (obs.cy - obs.hy < -halfSize)
        {
            obs.cy = -halfSize + obs.hy;
            obs.vy = std::abs(obs.vy);
        }
        if (obs.cy + obs.hy > halfSize)
        {
            obs.cy = halfSize - obs.hy;
            obs.vy = -std::abs(obs.vy);
        }
        if (obs.cz - obs.hz < -halfSize)
        {
            obs.cz = -halfSize + obs.hz;
            obs.vz = std::abs(obs.vz);
        }
        if (obs.cz + obs.hz > halfSize)
        {
            obs.cz = halfSize - obs.hz;
            obs.vz = -std::abs(obs.vz);
        }
    }

    // Elastic AABB-vs-AABB collision between obstacles
    for (size_t i = 0; i < simState.obstacles.size(); i++)
    {
        for (size_t j = i + 1; j < simState.obstacles.size(); j++)
        {
            Obstacle &a = simState.obstacles[i];
            Obstacle &b = simState.obstacles[j];

            float overlapX = (a.hx + b.hx) - std::abs(a.cx - b.cx);
            float overlapY = (a.hy + b.hy) - std::abs(a.cy - b.cy);
            float overlapZ = (a.hz + b.hz) - std::abs(a.cz - b.cz);

            if (overlapX > 0.0f && overlapY > 0.0f && overlapZ > 0.0f)
            {
                if (overlapX <= overlapY && overlapX <= overlapZ)
                {
                    float dir = (a.cx < b.cx) ? -1.0f : 1.0f;
                    a.cx += dir * overlapX * 0.5f;
                    b.cx -= dir * overlapX * 0.5f;
                    std::swap(a.vx, b.vx);
                }
                else if (overlapY <= overlapZ)
                {
                    float dir = (a.cy < b.cy) ? -1.0f : 1.0f;
                    a.cy += dir * overlapY * 0.5f;
                    b.cy -= dir * overlapY * 0.5f;
                    std::swap(a.vy, b.vy);
                }
                else
                {
                    float dir = (a.cz < b.cz) ? -1.0f : 1.0f;
                    a.cz += dir * overlapZ * 0.5f;
                    b.cz -= dir * overlapZ * 0.5f;
                    std::swap(a.vz, b.vz);
                }
            }
        }
    }
}

void spawnParticles(SimulationState &simState)
{
    for (int i = 0; i < partPerFrame; i++)
    {
        if (simState.particlesSpawned < (int)simState.particles.size())
        {
            float offsetRange = cellSize * 0.3f;
            simState.particles[simState.particlesSpawned].x = spawnX + ((rand() % 100) / 100.0f - 0.5f) * offsetRange;
            simState.particles[simState.particlesSpawned].y = simState.spawnY;
            simState.particles[simState.particlesSpawned].z = spawnZ + ((rand() % 100) / 100.0f - 0.5f) * offsetRange;

            simState.particles[simState.particlesSpawned].vx = ((rand() % 100) / 100.0f - 0.5f) * 0.2f;
            simState.particles[simState.particlesSpawned].vy = -0.1f;
            simState.particles[simState.particlesSpawned].vz = ((rand() % 100) / 100.0f - 0.5f) * 0.2f;

            simState.particles[simState.particlesSpawned].active = true;
            simState.particlesSpawned++;
        }
    }
}

void buildSpatialGrid(SimulationState &simState)
{
    int gridSize = gridWidth * gridWidth * gridWidth;
    simState.grid.clear();
    simState.grid.resize(gridSize);

    for (size_t i = 0; i < simState.particles.size(); i++)
    {
        if (!simState.particles[i].active)
            continue;

        int gx = gridIndex(simState.particles[i].x);
        int gy = gridIndex(simState.particles[i].y);
        int gz = gridIndex(simState.particles[i].z);

        gx = std::max(0, std::min(gridWidth - 1, gx));
        gy = std::max(0, std::min(gridWidth - 1, gy));
        gz = std::max(0, std::min(gridWidth - 1, gz));

        int hash = gridHash(gx, gy, gz) % gridSize;
        if (hash < 0)
            hash += gridSize;
        simState.grid[hash].push_back((int)i);
    }
}

void findNeighbors(SimulationState &simState, size_t particleIdx, std::vector<int> &neighbors)
{
    neighbors.clear();
    Particle &p = simState.particles[particleIdx];

    int gx = gridIndex(p.x);
    int gy = gridIndex(p.y);
    int gz = gridIndex(p.z);

    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dz = -1; dz <= 1; dz++)
            {
                int cx = gx + dx;
                int cy = gy + dy;
                int cz = gz + dz;

                if (cx < 0 || cx >= gridWidth ||
                    cy < 0 || cy >= gridWidth ||
                    cz < 0 || cz >= gridWidth)
                    continue;

                int hash = gridHash(cx, cy, cz) % simState.grid.size();
                if (hash < 0)
                    hash += simState.grid.size();

                for (int idx : simState.grid[hash])
                {
                    if ((size_t)idx == particleIdx)
                        continue;

                    Particle &neighbor = simState.particles[idx];
                    float dx = p.x - neighbor.x;
                    float dy = p.y - neighbor.y;
                    float dz = p.z - neighbor.z;
                    float distSq = dx * dx + dy * dy + dz * dz;

                    if (distSq < cellSize * cellSize)
                    {
                        neighbors.push_back(idx);
                    }
                }
            }
        }
    }
}

void computeDensityPressure(SimulationState &simState)
{
    std::vector<int> neighbors;

    for (size_t i = 0; i < simState.particles.size(); i++)
    {
        if (!simState.particles[i].active)
            continue;

        Particle &p = simState.particles[i];
        p.density = 0.0f;

        findNeighbors(simState, i, neighbors);

        for (int j : neighbors)
        {
            Particle &neighbor = simState.particles[j];
            float dx = p.x - neighbor.x;
            float dy = p.y - neighbor.y;
            float dz = p.z - neighbor.z;
            float distSq = dx * dx + dy * dy + dz * dz;

            if (distSq < cellSize * cellSize)
            {
                p.density += MASS * POLY6 * pow(cellSize * cellSize - distSq, 3);
            }
        }

        p.density += MASS * POLY6 * pow(cellSize * cellSize, 3);
        p.pressure = GAS_CONSTANT * std::max(0.0f, p.density - REST_DENISITY);
    }
}

void computeForces(SimulationState &simState)
{
    std::vector<int> neighbors;

    for (size_t i = 0; i < simState.particles.size(); i++)
    {
        if (!simState.particles[i].active)
            continue;

        Particle &p = simState.particles[i];
        float fpress_x = 0.0f, fpress_y = 0.0f, fpress_z = 0.0f;
        float fvisc_x = 0.0f, fvisc_y = 0.0f, fvisc_z = 0.0f;

        findNeighbors(simState, i, neighbors);

        for (int j : neighbors)
        {
            Particle &neighbor = simState.particles[j];
            float dx = p.x - neighbor.x;
            float dy = p.y - neighbor.y;
            float dz = p.z - neighbor.z;
            float dist = sqrt(dx * dx + dy * dy + dz * dz);

            if (dist > 0.0001f && dist < cellSize)
            {
                float nx = dx / dist;
                float ny = dy / dist;
                float nz = dz / dist;

                float pressureTerm = -MASS * (p.pressure + neighbor.pressure) / (2.0f * neighbor.density);
                float spiky = SPIKY_GRAD * pow(cellSize - dist, 2);

                fpress_x += pressureTerm * spiky * nx;
                fpress_y += pressureTerm * spiky * ny;
                fpress_z += pressureTerm * spiky * nz;

                float viscLap = VISC_LAP * (cellSize - dist);
                fvisc_x += VISCOSITY * MASS * (neighbor.vx - p.vx) / neighbor.density * viscLap;
                fvisc_y += VISCOSITY * MASS * (neighbor.vy - p.vy) / neighbor.density * viscLap;
                fvisc_z += VISCOSITY * MASS * (neighbor.vz - p.vz) / neighbor.density * viscLap;
            }
        }

        p.fx = fpress_x + fvisc_x;
        p.fy = fpress_y + fvisc_y + (MASS * gravity);
        p.fz = fpress_z + fvisc_z;
    }
}

void resolveCollisions(SimulationState &simState)
{
    float minDist = particleRad * 2.0f;
    float minDistSq = minDist * minDist;
    std::vector<int> neighbors;

    for (size_t i = 0; i < simState.particles.size(); i++)
    {
        if (!simState.particles[i].active)
            continue;

        Particle &p = simState.particles[i];
        findNeighbors(simState, i, neighbors);

        for (int j : neighbors)
        {
            if ((size_t)j <= i)
                continue;

            Particle &other = simState.particles[j];
            float dx = p.x - other.x;
            float dy = p.y - other.y;
            float dz = p.z - other.z;
            float distSq = dx * dx + dy * dy + dz * dz;

            if (distSq < minDistSq && distSq > 0.0001f)
            {
                float dist = sqrt(distSq);
                float overlap = (minDist - dist) * 0.5f;

                float nx = dx / dist;
                float ny = dy / dist;
                float nz = dz / dist;

                p.x += nx * overlap;
                p.y += ny * overlap;
                p.z += nz * overlap;

                other.x -= nx * overlap;
                other.y -= ny * overlap;
                other.z -= nz * overlap;
            }
        }
    }
}

void updatePhysics(SimulationState &simState)
{
    float halfSize = boxsize / 2.0f;

    updateObstacles(simState);
    spawnParticles(simState);
    buildSpatialGrid(simState);
    computeDensityPressure(simState);
    computeForces(simState);

    for (auto &p : simState.particles)
    {
        if (!p.active)
            continue;

        if (p.density < 0.01f)
            p.density = REST_DENISITY;

        float ax = p.fx / p.density;
        float ay = p.fy / p.density;
        float az = p.fz / p.density;

        p.vx += timeStep * ax;
        p.vy += timeStep * ay;
        p.vz += timeStep * az;

        p.x += timeStep * p.vx;
        p.y += timeStep * p.vy;
        p.z += timeStep * p.vz;

        if (p.x - particleRad < -halfSize)
        {
            p.x = -halfSize + particleRad;
            p.vx *= -damping;
        }
        else if (p.x + particleRad > halfSize)
        {
            p.x = halfSize - particleRad;
            p.vx *= -damping;
        }

        if (p.y - particleRad < -halfSize)
        {
            p.y = -halfSize + particleRad;
            p.vy *= -damping;
        }
        else if (p.y + particleRad > halfSize)
        {
            p.y = halfSize - particleRad;
            p.vy *= -damping;
        }

        if (p.z - particleRad < -halfSize)
        {
            p.z = -halfSize + particleRad;
            p.vz *= -damping;
        }
        else if (p.z + particleRad > halfSize)
        {
            p.z = halfSize - particleRad;
            p.vz *= -damping;
        }

        // check collision against each static obstacle
        for (const auto &obs : simState.obstacles)
        {
            float closestX = std::max(obs.cx - obs.hx, std::min(p.x, obs.cx + obs.hx));
            float closestY = std::max(obs.cy - obs.hy, std::min(p.y, obs.cy + obs.hy));
            float closestZ = std::max(obs.cz - obs.hz, std::min(p.z, obs.cz + obs.hz));

            float dx = p.x - closestX;
            float dy = p.y - closestY;
            float dz = p.z - closestZ;
            float distSq = dx * dx + dy * dy + dz * dz;

            if (distSq < particleRad * particleRad)
            {
                float nx, ny, nz;
                float penetration;

                if (distSq < 0.0001f)
                {

                    float overlapX = obs.hx - std::abs(p.x - obs.cx);
                    float overlapY = obs.hy - std::abs(p.y - obs.cy);
                    float overlapZ = obs.hz - std::abs(p.z - obs.cz);

                    if (overlapX <= overlapY && overlapX <= overlapZ)
                    {
                        nx = (p.x < obs.cx) ? -1.0f : 1.0f;
                        ny = nz = 0.0f;
                        penetration = overlapX + particleRad;
                    }
                    else if (overlapY <= overlapZ)
                    {
                        ny = (p.y < obs.cy) ? -1.0f : 1.0f;
                        nx = nz = 0.0f;
                        penetration = overlapY + particleRad;
                    }
                    else
                    {
                        nz = (p.z < obs.cz) ? -1.0f : 1.0f;
                        nx = ny = 0.0f;
                        penetration = overlapZ + particleRad;
                    }
                }
                else
                // normalize dx/dy/dz
                // move the particle out of the obstacle
                // Velocity reflection
                {

                    float dist = std::sqrt(distSq);
                    nx = dx / dist;
                    ny = dy / dist;
                    nz = dz / dist;
                    penetration = particleRad - dist;
                }

                p.x += nx * penetration;
                p.y += ny * penetration;
                p.z += nz * penetration;

                // velocity reflection
                float vdotn = p.vx * nx + p.vy * ny + p.vz * nz;
                if (vdotn < 0.0f)
                {
                    p.vx -= (1.0f + damping) * vdotn * nx;
                    p.vy -= (1.0f + damping) * vdotn * ny;
                    p.vz -= (1.0f + damping) * vdotn * nz;
                }
            }
        }
    }

    resolveCollisions(simState);
}