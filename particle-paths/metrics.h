#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <limits>

enum class Scenario
{
    PARTICLES_ONLY = 0,
    STATIC_OBSTACLES = 1,
    DYNAMIC_OBSTACLES = 2
};

inline const char *scenarioName(Scenario s)
{
    switch (s)
    {
    case Scenario::PARTICLES_ONLY:
        return "particles_only";
    case Scenario::STATIC_OBSTACLES:
        return "static_obstacles";
    case Scenario::DYNAMIC_OBSTACLES:
        return "dynamic_obstacles";
    default:
        return "unknown";
    }
}

inline const char *scenarioLabel(Scenario s)
{
    switch (s)
    {
    case Scenario::PARTICLES_ONLY:
        return "1: Particles Only";
    case Scenario::STATIC_OBSTACLES:
        return "2: Static Obstacles";
    case Scenario::DYNAMIC_OBSTACLES:
        return "3: Dynamic Obstacles";
    default:
        return "Unknown";
    }
}

struct ReplanEvent
{
    float timestamp;
    long long planTimeUs;
    float pathLength;
    int pathNodes;
    float pathDivergence;
    bool pathFound;
};

struct RunResult
{
    Scenario scenario;
    bool useRRTStar;
    int runNumber;
    bool success;
    float timeToGoal;
    int totalReplans;
    int failedReplans;
    long long avgPlanTimeUs;
    long long minPlanTimeUs;
    long long maxPlanTimeUs;
    float avgPathLength;
    float avgPathNodes;
    float avgPathDivergence;
    int particleCount;
};

inline RunResult compileRunResult(
    Scenario scenario, bool useRRTStar, int runNumber,
    bool success, float timeToGoal,
    const std::vector<ReplanEvent> &log,
    int particleCount = 0)
{
    RunResult r{};
    r.scenario = scenario;
    r.useRRTStar = useRRTStar;
    r.runNumber = runNumber;
    r.success = success;
    r.timeToGoal = success ? timeToGoal : -1.0f;
    r.totalReplans = (int)log.size();

    if (log.empty())
    {
        r.minPlanTimeUs = 0;
        r.maxPlanTimeUs = 0;
        return r;
    }

    long long sumT = 0;
    long long minT = std::numeric_limits<long long>::max();
    long long maxT = 0;
    float sumLen = 0, sumNodes = 0, sumDiv = 0;
    int failed = 0;

    for (const auto &e : log)
    {
        sumT += e.planTimeUs;
        if (e.planTimeUs < minT)
            minT = e.planTimeUs;
        if (e.planTimeUs > maxT)
            maxT = e.planTimeUs;
        sumLen += e.pathLength;
        sumNodes += (float)e.pathNodes;
        sumDiv += e.pathDivergence;
        if (!e.pathFound)
            failed++;
    }

    int n = (int)log.size();
    r.avgPlanTimeUs = sumT / (long long)n;
    r.minPlanTimeUs = minT;
    r.maxPlanTimeUs = maxT;
    r.avgPathLength = sumLen / (float)n;
    r.avgPathNodes = sumNodes / (float)n;
    r.avgPathDivergence = sumDiv / (float)n;
    r.failedReplans = failed;
    r.particleCount = particleCount;
    return r;
}

inline void appendRunResultCSV(const std::string &filePath, const RunResult &r)
{
    // Check if file is empty
    bool writeHeader = false;
    {
        std::ifstream check(filePath);
        writeHeader = (!check.good() ||
                       check.peek() == std::ifstream::traits_type::eof());
    }

    std::ofstream f(filePath, std::ios::app);
    if (writeHeader)
        f << "scenario,algorithm,run_number,success,time_to_goal_s,"
          << "total_replans,failed_replans,"
          << "avg_plan_time_us,min_plan_time_us,max_plan_time_us,"
          << "avg_path_length,avg_path_nodes,avg_path_divergence,particle_count\n";

    f << scenarioName(r.scenario) << ","
      << (r.useRRTStar ? "RRTstar" : "Astar") << ","
      << r.runNumber << ","
      << (r.success ? 1 : 0) << ","
      << r.timeToGoal << ","
      << r.totalReplans << ","
      << r.failedReplans << ","
      << r.avgPlanTimeUs << ","
      << r.minPlanTimeUs << ","
      << r.maxPlanTimeUs << ","
      << r.avgPathLength << ","
      << r.avgPathNodes << ","
      << r.avgPathDivergence << ","
      << r.particleCount << "\n";
    f.flush();
}

#endif
