#include "pathFinding.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <numeric>

int pathGridDim = 25;

static float computePathLength(const std::vector<std::vector<float>> &path)
{
    float len = 0.0f;
    for (size_t i = 1; i < path.size(); i++)
    {
        float dx = path[i][0] - path[i - 1][0];
        float dy = path[i][1] - path[i - 1][1];
        float dz = path[i][2] - path[i - 1][2];
        len += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    return len;
}

static float computePathDivergence(
    const std::vector<std::vector<float>> &oldPath,
    const std::vector<std::vector<float>> &newPath)
{
    if (oldPath.empty() || newPath.empty())
        return 0.0f;
    float total = 0.0f;
    for (const auto &wp : oldPath)
    {
        float minD = std::numeric_limits<float>::max();
        for (const auto &nwp : newPath)
        {
            float dx = wp[0] - nwp[0];
            float dy = wp[1] - nwp[1];
            float dz = wp[2] - nwp[2];
            float d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d < minD)
                minD = d;
        }
        total += minD;
    }
    return total / (float)oldPath.size();
}

void worldToPathgrid(float x, float y, float z, int &gx, int &gy, int &gz)
{
    float halfSize = boxsize / 2.0f;
    gx = (int)((x + halfSize) / pathCellSize);
    gy = (int)((y + halfSize) / pathCellSize);
    gz = (int)((z + halfSize) / pathCellSize);
}

void pathGridToWorld(int gx, int gy, int gz, float &x, float &y, float &z)
{
    float halfSize = boxsize / 2.0f;
    x = -halfSize + (gx + 0.5f) * pathCellSize;
    y = -halfSize + (gy + 0.5f) * pathCellSize;
    z = -halfSize + (gz + 0.5f) * pathCellSize;
}

bool isOccupied(const SimulationState &simState, int gx, int gy, int gz)
{
    float wx, wy, wz;
    pathGridToWorld(gx, gy, gz, wx, wy, wz);

    for (const auto &obs : simState.obstacles)
    {
        if (wx >= obs.cx - obs.hx - pathCellSize && wx <= obs.cx + obs.hx + pathCellSize &&
            wy >= obs.cy - obs.hy - pathCellSize && wy <= obs.cy + obs.hy + pathCellSize &&
            wz >= obs.cz - obs.hz - pathCellSize && wz <= obs.cz + obs.hz + pathCellSize)
            return true;
    }

    float checkRadius = pathCellSize * 0.7f;
    float checkRadSq = checkRadius * checkRadius;
    int gridSize = (int)simState.grid.size();

    if (gridSize > 0)
    {
        int sgx = gridIndex(wx);
        int sgy = gridIndex(wy);
        int sgz = gridIndex(wz);

        for (int ddx = -1; ddx <= 1; ddx++)
        {
            for (int ddy = -1; ddy <= 1; ddy++)
            {
                for (int ddz = -1; ddz <= 1; ddz++)
                {
                    int cx = sgx + ddx, cy = sgy + ddy, cz = sgz + ddz;
                    if (cx < 0 || cx >= gridWidth ||
                        cy < 0 || cy >= gridWidth ||
                        cz < 0 || cz >= gridWidth)
                        continue;

                    int hash = gridHash(cx, cy, cz) % gridSize;
                    if (hash < 0)
                        hash += gridSize;

                    for (int idx : simState.grid[hash])
                    {
                        if (!simState.particles[idx].active)
                            continue;
                        if (idx == simState.robotParticle ||
                            idx == simState.goalParticle)
                            continue;
                        float dx = simState.particles[idx].x - wx;
                        float dy = simState.particles[idx].y - wy;
                        float dz = simState.particles[idx].z - wz;
                        if (dx * dx + dy * dy + dz * dz < checkRadSq)
                            return true;
                    }
                }
            }
        }
    }

    return false;
}

bool isOccupiedWorld(const SimulationState &simState, float wx, float wy, float wz)
{
    float halfSize = boxsize / 2.0f;
    if (wx < -halfSize || wx > halfSize ||
        wy < -halfSize || wy > halfSize ||
        wz < -halfSize || wz > halfSize)
    {
        return true;
    }

    for (const auto &obs : simState.obstacles)
    {
        if (obs.contains(wx, wy, wz))
            return true;
    }

    float checkRadius = particleRad * 2.0f;
    float checkRadSq = checkRadius * checkRadius;
    int gridSize = (int)simState.grid.size();

    if (gridSize > 0)
    {
        int sgx = gridIndex(wx);
        int sgy = gridIndex(wy);
        int sgz = gridIndex(wz);

        for (int ddx = -1; ddx <= 1; ddx++)
        {
            for (int ddy = -1; ddy <= 1; ddy++)
            {
                for (int ddz = -1; ddz <= 1; ddz++)
                {
                    int cx = sgx + ddx, cy = sgy + ddy, cz = sgz + ddz;
                    if (cx < 0 || cx >= gridWidth ||
                        cy < 0 || cy >= gridWidth ||
                        cz < 0 || cz >= gridWidth)
                        continue;

                    int hash = gridHash(cx, cy, cz) % gridSize;
                    if (hash < 0)
                        hash += gridSize;

                    for (int idx : simState.grid[hash])
                    {
                        if (!simState.particles[idx].active)
                            continue;
                        if (idx == simState.robotParticle ||
                            idx == simState.goalParticle)
                            continue;
                        float dx = simState.particles[idx].x - wx;
                        float dy = simState.particles[idx].y - wy;
                        float dz = simState.particles[idx].z - wz;
                        if (dx * dx + dy * dy + dz * dz < checkRadSq)
                            return true;
                    }
                }
            }
        }
    }

    return false;
}

float heuristic(int x1, int y1, int z1, int x2, int y2, int z2)
{
    return sqrt((x1 - x2) * (x1 - x2) +
                (y1 - y2) * (y1 - y2) +
                (z1 - z2) * (z1 - z2));
}

bool findPath(const SimulationState &simState, int startX, int startY, int startZ,
              int goalX, int goalY, int goalZ, std::vector<std::vector<float>> &path)
{
    std::priority_queue<pathNode, std::vector<pathNode>, std::greater<pathNode>> openSet;
    std::unordered_map<std::tuple<int, int, int>, std::tuple<int, int, int>, Vec3Hash> cameFrom;
    std::unordered_map<std::tuple<int, int, int>, float, Vec3Hash> gScores;
    std::unordered_set<std::tuple<int, int, int>, Vec3Hash> closedSet;

    pathNode startNode(startX, startY, startZ, 0.0f, heuristic(startX, startY, startZ, goalX, goalY, goalZ));
    openSet.push(startNode);
    gScores[std::make_tuple(startX, startY, startZ)] = 0.0f;

    while (!openSet.empty())
    {
        pathNode current = openSet.top();
        openSet.pop();

        auto currentTuple = std::make_tuple(current.x, current.y, current.z);
        closedSet.insert(currentTuple);

        if (current.x == goalX && current.y == goalY && current.z == goalZ)
        {
            path.clear();
            path.push_back({current.x * pathCellSize - boxsize / 2.0f + pathCellSize / 2.0f,
                            current.y * pathCellSize - boxsize / 2.0f + pathCellSize / 2.0f,
                            current.z * pathCellSize - boxsize / 2.0f + pathCellSize / 2.0f});

            while (cameFrom.find(currentTuple) != cameFrom.end())
            {
                currentTuple = cameFrom[currentTuple];
                float wx = std::get<0>(currentTuple) * pathCellSize - boxsize / 2.0f + pathCellSize / 2.0f;
                float wy = std::get<1>(currentTuple) * pathCellSize - boxsize / 2.0f + pathCellSize / 2.0f;
                float wz = std::get<2>(currentTuple) * pathCellSize - boxsize / 2.0f + pathCellSize / 2.0f;
                path.push_back({wx, wy, wz});
            }

            std::reverse(path.begin(), path.end());
            return true;
        }

        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dz = -1; dz <= 1; dz++)
                {
                    int nx = current.x + dx;
                    int ny = current.y + dy;
                    int nz = current.z + dz;

                    if (nx < 0 || nx >= pathGridDim ||
                        ny < 0 || ny >= pathGridDim ||
                        nz < 0 || nz >= pathGridDim)
                        continue;

                    auto neighborTuple = std::make_tuple(nx, ny, nz);
                    if (closedSet.find(neighborTuple) != closedSet.end())
                        continue;

                    if (!(nx == goalX && ny == goalY && nz == goalZ) && isOccupied(simState, nx, ny, nz))
                        continue;

                    float moveCost = sqrt(dx * dx + dy * dy + dz * dz);
                    float tentativeG = gScores[currentTuple] + moveCost;

                    if (gScores.find(neighborTuple) == gScores.end() || tentativeG < gScores[neighborTuple])
                    {
                        gScores[neighborTuple] = tentativeG;
                        cameFrom[neighborTuple] = currentTuple;

                        float h = heuristic(nx, ny, nz, goalX, goalY, goalZ);
                        pathNode neighborNode(nx, ny, nz, tentativeG, h);
                        openSet.push(neighborNode);
                    }
                }
            }
        }
    }
    return false;
}

float distance3D(float x1, float y1, float z1, float x2, float y2, float z2)
{
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

bool isPathClear(const SimulationState &simState, float x1, float y1, float z1,
                 float x2, float y2, float z2)
{
    float dist = distance3D(x1, y1, z1, x2, y2, z2);
    int steps = (int)(dist / (particleRad * 2.0f)) + 1;

    for (int i = 0; i <= steps; i++)
    {
        float t = (float)i / (float)steps;
        float wx = x1 + t * (x2 - x1);
        float wy = y1 + t * (y2 - y1);
        float wz = z1 + t * (z2 - z1);

        if (isOccupiedWorld(simState, wx, wy, wz))
        {
            return false;
        }
    }

    return true;
}

float randomFloat(float min, float max)
{
    return min + (float)rand() / (float)RAND_MAX * (max - min);
}

static void smoothPath(const SimulationState &simState,
                       std::vector<std::vector<float>> &path)
{
    if (path.size() < 3)
        return;

    size_t i = 0;
    while (i + 2 < path.size())
    {
        size_t j = path.size() - 1;
        while (j > i + 1)
        {
            if (isPathClear(simState,
                            path[i][0], path[i][1], path[i][2],
                            path[j][0], path[j][1], path[j][2]))
            {
                path.erase(path.begin() + i + 1, path.begin() + j);
                break;
            }
            j--;
        }
        i++;
    }
}

static int buildKDTree(const std::vector<RRTNode> &rrtTree,
                       std::vector<int> &indices,
                       std::vector<KDNode> &kd,
                       int start, int end, int depth)
{
    if (start >= end)
        return -1;

    int axis = depth % 3;
    int mid = (start + end) / 2;

    std::nth_element(indices.begin() + start,
                     indices.begin() + mid,
                     indices.begin() + end,
                     [&](int a, int b)
                     {
                         if (axis == 0)
                             return rrtTree[a].x < rrtTree[b].x;
                         if (axis == 1)
                             return rrtTree[a].y < rrtTree[b].y;
                         return rrtTree[a].z < rrtTree[b].z;
                     });

    int rrtIndex = indices[mid];
    int kdIndex = (int)kd.size();
    kd.push_back({rrtTree[rrtIndex].x, rrtTree[rrtIndex].y, rrtTree[rrtIndex].z,
                  rrtIndex, -1, -1});

    kd[kdIndex].left = buildKDTree(rrtTree, indices, kd, start, mid, depth + 1);
    kd[kdIndex].right = buildKDTree(rrtTree, indices, kd, mid + 1, end, depth + 1);
    return kdIndex;
}

static void kdNearestSearch(const std::vector<KDNode> &kd, int nodeIndex,
                            float qx, float qy, float qz, int depth,
                            int &bestRRTIndex, float &bestDistSq)
{
    if (nodeIndex < 0)
        return;
    const KDNode &node = kd[nodeIndex];

    float dx = node.x - qx, dy = node.y - qy, dz = node.z - qz;
    float distSq = dx * dx + dy * dy + dz * dz;
    if (distSq < bestDistSq)
    {
        bestDistSq = distSq;
        bestRRTIndex = node.treeIdx;
    }

    int axis = depth % 3;
    float axisDelta = (axis == 0)   ? (qx - node.x)
                      : (axis == 1) ? (qy - node.y)
                                    : (qz - node.z);

    int near = (axisDelta <= 0) ? node.left : node.right;
    int far = (axisDelta <= 0) ? node.right : node.left;

    kdNearestSearch(kd, near, qx, qy, qz, depth + 1, bestRRTIndex, bestDistSq);

    if (axisDelta * axisDelta < bestDistSq)
        kdNearestSearch(kd, far, qx, qy, qz, depth + 1, bestRRTIndex, bestDistSq);
}

bool findPathRRTStar(const SimulationState &simState,
                     float startX, float startY, float startZ,
                     float goalX, float goalY, float goalZ,
                     std::vector<std::vector<float>> &path)
{
    const int maxIterations = 2000;
    const float stepSize = 0.3f;
    const float goalRadius = 0.3f;
    const float rewireRadius = 0.6f;
    const float halfSize = boxsize / 2.0f;

    std::vector<RRTNode> tree;
    tree.push_back(RRTNode(startX, startY, startZ, 0.0f, -1));

    const int REBUILD_INTERVAL = 50;
    std::vector<KDNode> kd;
    std::vector<int> kdIndices;
    int kdRoot = -1;
    int kdBuiltUpTo = 0;

    auto rebuildKD = [&]()
    {
        kd.clear();
        kd.reserve(tree.size());
        kdIndices.resize(tree.size());
        std::iota(kdIndices.begin(), kdIndices.end(), 0);
        kdRoot = buildKDTree(tree, kdIndices, kd, 0, (int)tree.size(), 0);
        kdBuiltUpTo = (int)tree.size();
    };
    rebuildKD();

    int goalNodeIndex = -1;

    for (int iter = 0; iter < maxIterations; iter++)
    {
        float randX, randY, randZ;

        if (rand() % 10 == 0)
        {
            randX = goalX;
            randY = goalY;
            randZ = goalZ;
        }
        else
        {
            randX = randomFloat(-halfSize, halfSize);
            randY = randomFloat(-halfSize, halfSize);
            randZ = randomFloat(-halfSize, halfSize);
        }

        int nearestIndex = 0;
        float nearestDistSq = std::numeric_limits<float>::max();
        kdNearestSearch(kd, kdRoot, randX, randY, randZ, 0, nearestIndex, nearestDistSq);
        for (int i = kdBuiltUpTo; i < (int)tree.size(); i++)
        {
            float dx = tree[i].x - randX, dy = tree[i].y - randY, dz = tree[i].z - randZ;
            float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < nearestDistSq)
            {
                nearestDistSq = d2;
                nearestIndex = i;
            }
        }
        float dx = randX - tree[nearestIndex].x;
        float dy = randY - tree[nearestIndex].y;
        float dz = randZ - tree[nearestIndex].z;
        float dist = sqrt(dx * dx + dy * dy + dz * dz);

        float newX, newY, newZ;
        if (dist > stepSize)
        {
            newX = tree[nearestIndex].x + dx / dist * stepSize;
            newY = tree[nearestIndex].y + dy / dist * stepSize;
            newZ = tree[nearestIndex].z + dz / dist * stepSize;
        }
        else
        {
            newX = randX;
            newY = randY;
            newZ = randZ;
        }

        if (isOccupiedWorld(simState, newX, newY, newZ))
            continue;

        if (!isPathClear(simState, tree[nearestIndex].x, tree[nearestIndex].y, tree[nearestIndex].z,
                         newX, newY, newZ))
            continue;

        std::vector<int> nearNodes;
        for (size_t i = 0; i < tree.size(); i++)
        {
            float d = distance3D(tree[i].x, tree[i].y, tree[i].z, newX, newY, newZ);
            if (d < rewireRadius)
            {
                nearNodes.push_back((int)i);
            }
        }

        int bestParent = nearestIndex;
        float bestCost = tree[nearestIndex].cost +
                         distance3D(tree[nearestIndex].x, tree[nearestIndex].y, tree[nearestIndex].z,
                                    newX, newY, newZ);

        for (int idx : nearNodes)
        {
            float newCost = tree[idx].cost +
                            distance3D(tree[idx].x, tree[idx].y, tree[idx].z, newX, newY, newZ);

            if (newCost < bestCost &&
                isPathClear(simState, tree[idx].x, tree[idx].y, tree[idx].z, newX, newY, newZ))
            {
                bestParent = idx;
                bestCost = newCost;
            }
        }

        int newNodeIndex = (int)tree.size();
        tree.push_back(RRTNode(newX, newY, newZ, bestCost, bestParent));

        for (int idx : nearNodes)
        {
            float rewireCost = bestCost +
                               distance3D(newX, newY, newZ, tree[idx].x, tree[idx].y, tree[idx].z);

            if (rewireCost < tree[idx].cost &&
                isPathClear(simState, newX, newY, newZ, tree[idx].x, tree[idx].y, tree[idx].z))
            {
                tree[idx].parent = newNodeIndex;
                tree[idx].cost = rewireCost;
            }
        }

        float distToGoal = distance3D(newX, newY, newZ, goalX, goalY, goalZ);
        if (distToGoal < goalRadius)
        {
            if (goalNodeIndex < 0 || tree[newNodeIndex].cost < tree[goalNodeIndex].cost)
            {
                goalNodeIndex = newNodeIndex;
            }
        }

        if ((int)tree.size() - kdBuiltUpTo >= REBUILD_INTERVAL)
            rebuildKD();
    }

    if (goalNodeIndex < 0)
    {
        return false;
    }

    path.clear();
    int currentIndex = goalNodeIndex;

    while (currentIndex >= 0)
    {
        path.push_back({tree[currentIndex].x, tree[currentIndex].y, tree[currentIndex].z});
        currentIndex = tree[currentIndex].parent;
    }

    std::reverse(path.begin(), path.end());
    return true;
}

void updatePath(SimulationState &simState)
{
    if (simState.robotParticle < 0 || simState.goalParticle < 0)
        return;

    if (!simState.particles[simState.robotParticle].active ||
        !simState.particles[simState.goalParticle].active)
        return;

    float startX = simState.particles[simState.robotParticle].x;
    float startY = simState.particles[simState.robotParticle].y;
    float startZ = simState.particles[simState.robotParticle].z;
    float goalX = simState.particles[simState.goalParticle].x;
    float goalY = simState.particles[simState.goalParticle].y;
    float goalZ = simState.particles[simState.goalParticle].z;

    std::vector<std::vector<float>> oldPath = simState.currentPath;
    auto t0 = std::chrono::high_resolution_clock::now();
    bool pathFound = false;

    if (simState.useRRTStar)
    {
        pathFound = findPathRRTStar(simState, startX, startY, startZ,
                                    goalX, goalY, goalZ, simState.currentPath);
    }
    else
    {
        int startGX, startGY, startGZ, goalGX, goalGY, goalGZ;
        worldToPathgrid(startX, startY, startZ, startGX, startGY, startGZ);
        worldToPathgrid(goalX, goalY, goalZ, goalGX, goalGY, goalGZ);
        pathFound = findPath(simState, startGX, startGY, startGZ,
                             goalGX, goalGY, goalGZ, simState.currentPath);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    long long planUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    if (!pathFound)
        simState.currentPath.clear();
    else
        smoothPath(simState, simState.currentPath);
    ReplanEvent ev{};
    ev.timestamp = simState.elapsedTime;
    ev.planTimeUs = planUs;
    ev.pathFound = pathFound;
    ev.pathLength = computePathLength(simState.currentPath);
    ev.pathNodes = (int)simState.currentPath.size();
    ev.pathDivergence = computePathDivergence(oldPath, simState.currentPath);
    simState.replanLog.push_back(ev);
}