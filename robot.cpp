#include "robot.h"
#include <cmath>
#include <iostream>
#include <cstdlib>

void initializeRobotAndGoal(SimulationState &simState)
{
    if (!simState.initialAgitationDone)
        return;

    if (simState.robotParticle >= 0 && simState.goalParticle >= 0)
        return;

    int numParticles = (int)simState.particles.size();
    float minSeparation = boxsize * 0.5f;

    int attempts = 0;
    int maxAttempts = 100;

    while (attempts < maxAttempts)
    {
        int robotIdx = rand() % numParticles;
        int goalIdx = rand() % numParticles;

        if (robotIdx == goalIdx)
        {
            attempts++;
            continue;
        }

        if (!simState.particles[robotIdx].active || !simState.particles[goalIdx].active)
        {
            attempts++;
            continue;
        }

        Particle &r = simState.particles[robotIdx];
        Particle &g = simState.particles[goalIdx];

        float dx = r.x - g.x;
        float dy = r.y - g.y;
        float dz = r.z - g.z;
        float dist = sqrt(dx * dx + dy * dy + dz * dz);

        if (dist >= minSeparation)
        {
            simState.robotParticle = robotIdx;
            simState.goalParticle = goalIdx;
            std::cout << "Robot initialized at particle index " << robotIdx << std::endl;
            std::cout << "Goal initialized at particle index " << goalIdx << std::endl;
            std::cout << "Initial separation: " << dist << std::endl;
            return;
        }

        attempts++;
    }

    simState.robotParticle = rand() % numParticles;
    do
    {
        simState.goalParticle = rand() % numParticles;
    } while (simState.goalParticle == simState.robotParticle);

    std::cout << "Robot initialized at particle index " << simState.robotParticle << " (fallback)" << std::endl;
    std::cout << "Goal initialized at particle index " << simState.goalParticle << " (fallback)" << std::endl;
    // aiv
}

void steerTowardsTarget(Particle &robot, float targetX, float targetY, float targetZ)
{
    float dx = targetX - robot.x;
    float dy = targetY - robot.y;
    float dz = targetZ - robot.z;

    float dist = sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < 0.001f)
        return;

    float ndx = dx / dist;
    float ndy = dy / dist;
    float ndz = dz / dist;

    float desiredVx = ndx * robotMaxSpeed;
    float desiredVy = ndy * robotMaxSpeed;
    float desiredVz = ndz * robotMaxSpeed;

    robot.vx += (desiredVx - robot.vx) * robotSteerForce * timeStep;
    robot.vy += (desiredVy - robot.vy) * robotSteerForce * timeStep;
    robot.vz += (desiredVz - robot.vz) * robotSteerForce * timeStep;
}

void updateRobotControl(SimulationState &simState)
{
    if (simState.robotParticle < 0 || simState.goalParticle < 0)
        return;

    if (!simState.particles[simState.robotParticle].active)
        return;

    Particle &robot = simState.particles[simState.robotParticle];
    Particle &goal = simState.particles[simState.goalParticle];

    float dx = robot.x - goal.x;
    float dy = robot.y - goal.y;
    float dz = robot.z - goal.z;
    float distToGoal = sqrt(dx * dx + dy * dy + dz * dz);

    if (distToGoal < particleRad * 4.0f)
    {
        simState.goalReached = true;
        std::cout << "Robot reached goal!" << std::endl;
        return;
    }

    if (simState.currentPath.empty())
        return;

    static int currentWaypoint = 0;
    static int lastPathCounter = -1;

    if (simState.pathUpdateCounter != lastPathCounter)
    {
        currentWaypoint = 0;
        lastPathCounter = simState.pathUpdateCounter;
    }

    if (currentWaypoint >= (int)simState.currentPath.size())
    {
        currentWaypoint = (int)simState.currentPath.size() - 1;
    }

    if (currentWaypoint < 0)
    {
        currentWaypoint = 0;
    }

    float targetX = simState.currentPath[currentWaypoint][0];
    float targetY = simState.currentPath[currentWaypoint][1];
    float targetZ = simState.currentPath[currentWaypoint][2];

    float dwx = robot.x - targetX;
    float dwy = robot.y - targetY;
    float dwz = robot.z - targetZ;
    float distToWaypoint = sqrt(dwx * dwx + dwy * dwy + dwz * dwz);

    if (distToWaypoint < waypointReachDist)
    {
        currentWaypoint++;
        if (currentWaypoint < (int)simState.currentPath.size())
        {
            targetX = simState.currentPath[currentWaypoint][0];
            targetY = simState.currentPath[currentWaypoint][1];
            targetZ = simState.currentPath[currentWaypoint][2];
        }
    }

    steerTowardsTarget(robot, targetX, targetY, targetZ);
}
