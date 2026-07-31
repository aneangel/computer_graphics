#ifndef ROBOT_H
#define ROBOT_H

#include "particle.h"

const float robotSteerForce = 50.0f;
const float robotMaxSpeed = 2.0f;
const float waypointReachDist = 0.3f;

void initializeRobotAndGoal(SimulationState &simState);
void updateRobotControl(SimulationState &simState);
void steerTowardsTarget(Particle &robot, float targetX, float targetY, float targetZ);

#endif
