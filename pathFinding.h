#ifndef PATHFINDING_H
#define PATHFINDING_H

#include "particle.h"
#include <vector>
#include <tuple>

const float pathCellSize = 0.2f;
extern int pathGridDim;

struct pathNode
{
    int x, y, z;
    float g, h, f;
    int parentX, parentY, parentZ;

    pathNode(int px, int py, int pz, float pg = 0, float ph = 0)
        : x(px), y(py), z(pz), g(pg), h(ph), f(pg + ph), parentX(-1), parentY(-1), parentZ(-1) {}

    bool operator>(const pathNode &other) const
    {
        return f > other.f;
    }
};

struct RRTNode
{
    float x, y, z;
    float cost;
    int parent;

    RRTNode(float px, float py, float pz, float c = 0.0f, int p = -1)
        : x(px), y(py), z(pz), cost(c), parent(p) {}
};

struct KDNode
{
    float x, y, z;
    int treeIdx;
    int left;
    int right;
};

struct Vec3Hash
{
    std::size_t operator()(const std::tuple<int, int, int> &v) const
    {
        auto h1 = std::hash<int>()(std::get<0>(v));
        auto h2 = std::hash<int>()(std::get<1>(v));
        auto h3 = std::hash<int>()(std::get<2>(v));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

void worldToPathgrid(float x, float u, float z, int &gx, int &gy, int &gz);
void pathGridToWorld(int gx, int gy, int gz, float &x, float &y, float &z);

bool isOccupied(const SimulationState &simState, int gx, int gy, int gz);
bool isOccupiedWorld(const SimulationState &simState, float wx, float wy, float wz);
float heuristic(int x1, int y1, int z1, int x2, int y2, int z2);

bool findPath(const SimulationState &simState, int startX, int startY, int startZ,
              int goalX, int goalY, int goalZ, std::vector<std::vector<float>> &path);

bool findPathRRTStar(const SimulationState &simState,
                     float startX, float startY, float startZ,
                     float goalX, float goalY, float goalZ,
                     std::vector<std::vector<float>> &path);

void updatePath(SimulationState &simState);

#endif
