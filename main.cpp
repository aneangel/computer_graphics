#include <GL/glew.h>
#include <GL/glut.h>

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

#include "particle.h"
#include "pathFinding.h"
#include "robot.h"
#include "gpu_renderer.h"
#include "metrics.h"
#ifdef USE_CUDA
#include "particle_cuda.h"
#endif

static const std::string CSV_FILE = "results.csv";
static int  g_maxRuns  = 0;
static bool g_headless = false;

void initScaleFromParticleCount(int n)
{
    float scale = std::cbrt((float)n / 1500.0f);
    boxsize = 5.0f * scale;
    gridWidth = std::max(35, (int)(boxsize / cellSize) + 2);
    pathGridDim = std::max(25, (int)(boxsize / pathCellSize));
}

const int WINDOW_WIDTH = 1500;
const int WINDOW_HEIGHT = 1300;

float cameraAngleX = 30.0f;
float cameraAngleY = 45.0f;
float cameraDistance = 10.0f;

bool mouseLeftDown = false;

static std::mutex g_pathMutex;
static std::atomic<bool> g_plannerRunning{false};
static std::vector<std::vector<float>> g_pendingPath;
static ReplanEvent g_pendingEvent{};
static std::atomic<bool> g_pathReady{false};

int mouseX = 0;
int mouseY = 0;

int frameCounter = 0;
// SimulationState simState;

void setupLighting()
{
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glEnable(GL_NORMALIZE);

    GLfloat lightPos0[] = {0.0f, 10.0f, 0.0f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);
    GLfloat ambient0[] = {0.15f, 0.15f, 0.15f, 1.0f};
    GLfloat diffuse0[] = {0.9f, 0.9f, 0.85f, 1.0f};
    GLfloat specular0[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient0);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse0);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular0);

    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0f);
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.02f);
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.005f);
    // GLfloat lightPosition[] = {0.0f, 5.0f, 0.0f, 1.0f};
    // glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);

    // GLfloat ambientLight[] = {0.3f, 0.3f, 0.3f, 1.0f};
    // glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);

    // GLfloat diffuseLight[] = {0.8f, 0.8f, 0.8f, 1.0f};
    // glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);

    // GLfloat specularLight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    // glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);

    // glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0f);
    // glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05f);
    // glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.01f);
    GLfloat lightPos1[] = {0.0f, -10.0f, 0.0f, 1.0f};
    glLightfv(GL_LIGHT1, GL_POSITION, lightPos1);
    GLfloat ambient1[] = {0.05f, 0.05f, 0.1f, 1.0f};
    GLfloat diffuse1[] = {0.2f, 0.2f, 0.35f, 1.0f};
    GLfloat specular1[] = {0.0f, 0.0f, 0.0f, 1.0f};
    glLightfv(GL_LIGHT1, GL_AMBIENT, ambient1);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse1);
    glLightfv(GL_LIGHT1, GL_SPECULAR, specular1);

    GLfloat globalAmbient[] = {0.2f, 0.2f, 0.25f, 1.0f};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, globalAmbient);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
}

void setMaterial(GLfloat r, GLfloat g, GLfloat b, GLfloat shininess)
{
    GLfloat materialAmbient[] = {r * 0.2f, g * 0.2f, b * 0.2f, 1.0f};
    GLfloat materialDiffuse[] = {r, g, b, 1.0f};
    GLfloat materialSpecular[] = {1.0f, 1.0f, 1.0f, 1.0f};

    glMaterialfv(GL_FRONT, GL_AMBIENT, materialAmbient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, materialDiffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, materialSpecular);
    glMaterialf(GL_FRONT, GL_SHININESS, shininess);
}

GLuint cubeDisplayList;

void buildCubeList()
{
    cubeDisplayList = glGenLists(1);
    glNewList(cubeDisplayList, GL_COMPILE);
    glDisable(GL_LIGHTING);
    glColor3f(0.0f, 0.0f, 0.0f);
    glLineWidth(1.0f);
    float halfSize = boxsize / 2.0f;

    glBegin(GL_LINES);
    // Bottom face
    glVertex3f(-halfSize, -halfSize, -halfSize);
    glVertex3f(halfSize, -halfSize, -halfSize);

    glVertex3f(halfSize, -halfSize, -halfSize);
    glVertex3f(halfSize, -halfSize, halfSize);

    glVertex3f(halfSize, -halfSize, halfSize);
    glVertex3f(-halfSize, -halfSize, halfSize);

    glVertex3f(-halfSize, -halfSize, halfSize);
    glVertex3f(-halfSize, -halfSize, -halfSize);

    // Top face
    glVertex3f(-halfSize, halfSize, -halfSize);
    glVertex3f(halfSize, halfSize, -halfSize);

    glVertex3f(halfSize, halfSize, -halfSize);
    glVertex3f(halfSize, halfSize, halfSize);

    glVertex3f(halfSize, halfSize, halfSize);
    glVertex3f(-halfSize, halfSize, halfSize);

    glVertex3f(-halfSize, halfSize, halfSize);
    glVertex3f(-halfSize, halfSize, -halfSize);

    // Vertical edges
    glVertex3f(-halfSize, -halfSize, -halfSize);
    glVertex3f(-halfSize, halfSize, -halfSize);

    glVertex3f(halfSize, -halfSize, -halfSize);
    glVertex3f(halfSize, halfSize, -halfSize);

    glVertex3f(halfSize, -halfSize, halfSize);
    glVertex3f(halfSize, halfSize, halfSize);

    glVertex3f(-halfSize, -halfSize, halfSize);
    glVertex3f(-halfSize, halfSize, halfSize);
    glEnd();
    glEnable(GL_LIGHTING);
    glEndList();
}

void drawWireframeCube()
{
    glCallList(cubeDisplayList);
    // glDisable(GL_LIGHTING);
    // glColor3f(0.0f, 0.0f, 0.0f);
    // glLineWidth(1.0f);

    // float halfSize = boxsize / 2.0f;

    //     glEnable(GL_LIGHTING);
}

void launchPlannerAsync()
{
    if (g_plannerRunning.exchange(true))
        return;

    SimulationState snap = simState;
    snap.replanLog.clear(); // only want the single new event from this call

    std::thread([snap = std::move(snap)]() mutable
                {
        updatePath(snap);
        {
            std::lock_guard<std::mutex> lk(g_pathMutex);
            g_pendingPath = snap.currentPath;
            g_pendingEvent = snap.replanLog.empty() ? ReplanEvent{} : snap.replanLog.back();
        }
        g_pathReady = true;
        g_plannerRunning = false; })
        .detach();
}

void drawObstacles()
{
    for (const auto &obs : simState.obstacles)
    {
        float x0 = obs.cx - obs.hx, x1 = obs.cx + obs.hx;
        float y0 = obs.cy - obs.hy, y1 = obs.cy + obs.hy;
        float z0 = obs.cz - obs.hz, z1 = obs.cz + obs.hz;

        glColor3f(0.85f, 0.1f, 0.1f);
        setMaterial(0.85f, 0.1f, 0.1f, 40.0f);

        glBegin(GL_QUADS);
        // -Y
        glNormal3f(0, -1, 0);
        glVertex3f(x0, y0, z0);
        glVertex3f(x1, y0, z0);
        glVertex3f(x1, y0, z1);
        glVertex3f(x0, y0, z1);
        // +Y
        glNormal3f(0, 1, 0);
        glVertex3f(x0, y1, z0);
        glVertex3f(x0, y1, z1);
        glVertex3f(x1, y1, z1);
        glVertex3f(x1, y1, z0);
        // -X
        glNormal3f(-1, 0, 0);
        glVertex3f(x0, y0, z0);
        glVertex3f(x0, y0, z1);
        glVertex3f(x0, y1, z1);
        glVertex3f(x0, y1, z0);
        // +X
        glNormal3f(1, 0, 0);
        glVertex3f(x1, y0, z0);
        glVertex3f(x1, y1, z0);
        glVertex3f(x1, y1, z1);
        glVertex3f(x1, y0, z1);
        // -Z
        glNormal3f(0, 0, -1);
        glVertex3f(x0, y0, z0);
        glVertex3f(x0, y1, z0);
        glVertex3f(x1, y1, z0);
        glVertex3f(x1, y0, z0);
        // +Z
        glNormal3f(0, 0, 1);
        glVertex3f(x0, y0, z1);
        glVertex3f(x1, y0, z1);
        glVertex3f(x1, y1, z1);
        glVertex3f(x0, y1, z1);
        glEnd();
    }
}

void drawPath()
{
    std::vector<std::vector<float>> path;
    {
        std::lock_guard<std::mutex> lk(g_pathMutex);
        path = simState.currentPath;
    }

    if (path.empty())
        return;

    glDisable(GL_LIGHTING);
    glColor3f(1.0f, 0.0f, 1.0f);
    glLineWidth(3.0f);

    glBegin(GL_LINE_STRIP);
    for (const auto &point : path)
    {
        glVertex3f(point[0], point[1], point[2]);
    }
    glEnd();

    glEnable(GL_LIGHTING);
}

void drawParticles()
{
    if (GPUinstance)
    {
        drawGPU(simState.particles, simState.robotParticle, simState.goalParticle);
        return;
    }

    float camX = cameraDistance * sin(cameraAngleY * M_PI / 180.0f) * cos(cameraAngleX * M_PI / 180.0f);
    float camY = cameraDistance * sin(cameraAngleX * M_PI / 180.0f);
    float camZ = cameraDistance * cos(cameraAngleY * M_PI / 180.0f) * cos(cameraAngleX * M_PI / 180.0f);

    for (size_t i = 0; i < simState.particles.size(); i++)
    {
        if (!simState.particles[i].active)
            continue;
        if ((int)i == simState.robotParticle || (int)i == simState.goalParticle)
            continue;

        float dx = simState.particles[i].x - camX;
        float dy = simState.particles[i].y - camY;
        float dz = simState.particles[i].z - camZ;

        float dist = sqrt(dx * dx + dy * dy + dz * dz);
        int slices = (dist < 5.0f) ? 8 : (dist < 10.0f) ? 5
                                                        : 3;
        //     continue;

        // if ((int)i == simState.robotParticle || (int)i == simState.goalParticle)
        //     continue;

        glPushMatrix();
        setMaterial(0.88f, 0.94f, 1.0f, 80.0f); // white/light blue
        glTranslatef(simState.particles[i].x, simState.particles[i].y, simState.particles[i].z);
        glutSolidSphere(particleRad, slices, slices);
        glPopMatrix();
    }

    float highlightRadius = particleRad;

    glDisable(GL_LIGHTING);

    if (simState.robotParticle >= 0 && simState.particles[simState.robotParticle].active)
    {
        glPushMatrix();
        glColor3f(1.0f, 1.0f, 1.0f); // white
        glTranslatef(simState.particles[simState.robotParticle].x,
                     simState.particles[simState.robotParticle].y,
                     simState.particles[simState.robotParticle].z);
        glutSolidSphere(highlightRadius, 16, 16);
        glPopMatrix();
    }

    if (simState.goalParticle >= 0 && simState.particles[simState.goalParticle].active)
    {
        glPushMatrix();
        glColor3f(1.0f, 0.0f, 0.0f);
        glTranslatef(simState.particles[simState.goalParticle].x,
                     simState.particles[simState.goalParticle].y,
                     simState.particles[simState.goalParticle].z);
        glutSolidSphere(highlightRadius, 16, 16);
        glPopMatrix();
    }

    glEnable(GL_LIGHTING);
}

void drawText(float x, float y, const char *text)
{
    glRasterPos2f(x, y);
    for (const char *c = text; *c; c++)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
    }
}

static char hudScenario[64], hudAlgorithm[64], hudParticles[64];
static char hudPath[64], hudTime[64], hudRecomp[64], hudPlanTime[64], hudRunNum[64];
static int hudUpdateCntr;

void drawHUD()
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (hudUpdateCntr++ % 10 == 0)
    {
        sprintf(hudScenario, "Scenario: %s", scenarioLabel(simState.currentScenario));
        sprintf(hudAlgorithm, "Algorithm: %s", simState.useRRTStar ? "RRT*" : "A*");
        sprintf(hudParticles, "Particles: %d", simState.particlesSpawned);
        sprintf(hudPath, "Path Steps: %d", (int)simState.currentPath.size());
        sprintf(hudTime, "Time: %.2f s", simState.elapsedTime);
        sprintf(hudRecomp, "Replans: %d", simState.pathUpdateCounter);
        sprintf(hudRunNum, "Run: %d", simState.runCount);

        if (!simState.replanLog.empty())
        {
            long long lastUs = simState.replanLog.back().planTimeUs;
            long long sumUs = 0;
            for (const auto &e : simState.replanLog)
                sumUs += e.planTimeUs;
            long long avgUs = sumUs / (long long)simState.replanLog.size();
            sprintf(hudPlanTime, "Plan: last=%lldus avg=%lldus", lastUs, avgUs);
        }
        else
            sprintf(hudPlanTime, "Plan: --");
    }

    int y = WINDOW_HEIGHT - 25;
    drawText(10, y, hudScenario);
    y -= 25;
    drawText(10, y, hudAlgorithm);
    y -= 25;
    drawText(10, y, hudParticles);
    y -= 25;
    drawText(10, y, hudTime);
    y -= 25;
    drawText(10, y, hudPath);
    y -= 25;
    drawText(10, y, hudRecomp);
    y -= 25;
    drawText(10, y, hudPlanTime);
    y -= 25;
    drawText(10, y, hudRunNum);

    // Controls (right side)
    drawText(WINDOW_WIDTH - 240, WINDOW_HEIGHT - 25, "[1/2/3] Scenario");
    drawText(WINDOW_WIDTH - 240, WINDOW_HEIGHT - 50, "[P] Toggle Algorithm");
    drawText(WINDOW_WIDTH - 240, WINDOW_HEIGHT - 75, "[R] Reset");
    drawText(WINDOW_WIDTH - 240, WINDOW_HEIGHT - 100, "[Space] Agitate");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float camX = cameraDistance * sin(cameraAngleY * M_PI / 180.0f) * cos(cameraAngleX * M_PI / 180.0f);
    float camY = cameraDistance * sin(cameraAngleX * M_PI / 180.0f);
    float camZ = cameraDistance * cos(cameraAngleY * M_PI / 180.0f) * cos(cameraAngleX * M_PI / 180.0f);

    gluLookAt(camX, camY, camZ, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    drawWireframeCube();
    drawObstacles();
    drawParticles();
    drawPath();
    drawHUD();

    glutSwapBuffers();
}

void reshape(int width, int height)
{
    if (height == 0)
        height = 1;

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float)width / (float)height, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
}

void agitateParticles()
{
    for (auto &p : simState.particles)
    {
        if (p.active)
        {
            p.vx += ((rand() % 100) / 100.0f - 0.5f) * 5.0f;
            p.vy += ((rand() % 100) / 100.0f) * 3.0f;
            p.vz += ((rand() % 100) / 100.0f - 0.5f) * 5.0f;
        }
    }
}

static void handleGoalReached(int &localFrameCounter)
{
    RunResult result = compileRunResult(
        simState.currentScenario,
        simState.useRRTStar,
        simState.runCount,
        true,
        simState.elapsedTime,
        simState.replanLog,
        simState.particlesSpawned);

    appendRunResultCSV(CSV_FILE, result);

    std::cout << "\n=== Run " << simState.runCount << " Complete ===" << std::endl;
    std::cout << "Scenario  : " << scenarioLabel(simState.currentScenario) << std::endl;
    std::cout << "Algorithm : " << (simState.useRRTStar ? "RRT*" : "A*") << std::endl;
    std::cout << "Time      : " << simState.elapsedTime << " s" << std::endl;
    std::cout << "Replans   : " << result.totalReplans
              << "  (failed: " << result.failedReplans << ")" << std::endl;
    std::cout << "Plan time : avg=" << result.avgPlanTimeUs
              << " min=" << result.minPlanTimeUs
              << " max=" << result.maxPlanTimeUs << " us" << std::endl;
    std::cout << "Path len  : " << result.avgPathLength << " (avg)" << std::endl;
    std::cout << "Divergence: " << result.avgPathDivergence << " (avg)" << std::endl;
    std::cout << "Saved to  : " << CSV_FILE << std::endl;
    std::cout << "===========================" << std::endl;

    if (g_maxRuns > 0 && simState.runCount >= g_maxRuns)
    {
        if (!g_headless)
            glutIdleFunc(nullptr);
#ifdef USE_CUDA
        cuda_cleanupPhysics();
#endif
        if (!g_headless)
            cleanupGPU();
        exit(0);
    }

    simState.reset();
    for (int i = 0; i < simState.targgetParticleCount; i++)
        simState.particles.push_back(Particle(0.0f, 0.0f, 0.0f));
    localFrameCounter = 0;
}

void runHeadless()
{
    int frameCounter = 0;

    while (true)
    {
        if (g_pathReady.exchange(false))
        {
            std::lock_guard<std::mutex> lk(g_pathMutex);
            if (g_pendingEvent.pathFound)
            {
                simState.currentPath = g_pendingPath;
                simState.pathUpdateCounter++;
            }
            simState.replanLog.push_back(g_pendingEvent);
        }

#ifdef USE_CUDA
        updateObstacles(simState);
        cuda_updatePhysics(simState);
#else
        updatePhysics(simState);
#endif
        initializeRobotAndGoal(simState);

        if (simState.particlesSpawned >= simState.targgetParticleCount && !simState.initialAgitationDone)
        {
            simState.settlingFrames++;
            if (simState.settlingFrames == 1)
                agitateParticles();
            if (simState.settlingFrames >= 200)
            {
                simState.initialAgitationDone = true;
                std::cout << "Particles settled, selecting robot and goal" << std::endl;
            }
        }

        updateRobotControl(simState);

        if (simState.robotParticle >= 0 && simState.goalParticle >= 0 && !simState.goalReached)
            simState.elapsedTime += timeStep;

        if (simState.goalReached)
            handleGoalReached(frameCounter);

        frameCounter++;
        if (frameCounter >= 10)
        {
            frameCounter = 0;
            launchPlannerAsync();
        }
    }
}

void idle()
{
    static auto lastTick = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - lastTick).count();
    if (dt < timeStep)
        return;
    lastTick = now;

    if (g_pathReady.exchange(false))
    {
        std::lock_guard<std::mutex> lk(g_pathMutex);
        if (g_pendingEvent.pathFound)
        {
            simState.currentPath = g_pendingPath;
            simState.pathUpdateCounter++;
        }
        simState.replanLog.push_back(g_pendingEvent);
    }

#ifdef USE_CUDA
    updateObstacles(simState);
    cuda_updatePhysics(simState);
#else
    updatePhysics(simState);
#endif
    initializeRobotAndGoal(simState);

    if (simState.particlesSpawned >= simState.targgetParticleCount && !simState.initialAgitationDone)
    {
        simState.settlingFrames++;

        if (simState.settlingFrames == 1)
        {
            agitateParticles();
            std::cout << "Initial agitation applied, waiting for particles to settle..." << std::endl;
        }

        if (simState.settlingFrames >= 200)
        {
            simState.initialAgitationDone = true;
            std::cout << "Particles settled, selecting robot and goal" << std::endl;
        }
    }

    updateRobotControl(simState);

    if (simState.robotParticle >= 0 && simState.goalParticle >= 0 && !simState.goalReached)
    {
        simState.elapsedTime += timeStep;
    }

    if (simState.goalReached)
        handleGoalReached(frameCounter);

    frameCounter++;
    if (frameCounter >= 10)
    {
        frameCounter = 0;
        launchPlannerAsync();
    }

    glutPostRedisplay();
}

void resetSimulation()
{
    simState.reset();
    for (int i = 0; i < simState.targgetParticleCount; i++)
        simState.particles.push_back(Particle(0.0f, 0.0f, 0.0f));
    frameCounter = 0;
}

void keyboard(unsigned char key, int x, int y)
{
    (void)x;
    (void)y;

    switch (key)
    {
    case 'q':
    case 'Q':
    case 27: // ESC
        glutIdleFunc(nullptr);
#ifdef USE_CUDA
        cuda_cleanupPhysics();
#endif
        cleanupGPU();
        exit(0);
        break;

    case 'w':
    case 'W':
        cameraAngleX += 5.0f;
        glutPostRedisplay();
        break;

    case 's':
    case 'S':
        cameraAngleX -= 5.0f;
        glutPostRedisplay();
        break;

    case 'a':
    case 'A':
        cameraAngleY -= 5.0f;
        glutPostRedisplay();
        break;

    case 'd':
    case 'D':
        cameraAngleY += 5.0f;
        glutPostRedisplay();
        break;

    case '+':
    case '=':
        cameraDistance -= 0.5f;
        if (cameraDistance < 1.0f)
            cameraDistance = 1.0f;
        glutPostRedisplay();
        break;

    case '-':
    case '_':
        cameraDistance += 0.5f;
        glutPostRedisplay();
        break;

    case 'r':
    case 'R':
        resetSimulation();
        std::cout << "Simulation reset (" << scenarioLabel(simState.currentScenario) << ")" << std::endl;
        break;

    case '1':
        simState.currentScenario = Scenario::PARTICLES_ONLY;
        resetSimulation();
        std::cout << "Scenario: Particles Only" << std::endl;
        break;

    case '2':
        simState.currentScenario = Scenario::STATIC_OBSTACLES;
        resetSimulation();
        std::cout << "Scenario: Static Obstacles" << std::endl;
        break;

    case '3':
        simState.currentScenario = Scenario::DYNAMIC_OBSTACLES;
        resetSimulation();
        std::cout << "Scenario: Dynamic Obstacles" << std::endl;
        break;

    case ' ':
        for (auto &p : simState.particles)
        {
            if (p.active)
            {
                p.vx += ((rand() % 100) / 100.0f - 0.5f) * 5.0f;
                p.vy += ((rand() % 100) / 100.0f) * 3.0f;
                p.vz += ((rand() % 100) / 100.0f - 0.5f) * 5.0f;
            }
        }
        std::cout << "Particles agitated" << std::endl;
        break;

    case 'p':
    case 'P':
        simState.useRRTStar = !simState.useRRTStar;
        std::cout << "Switched to: " << (simState.useRRTStar ? "RRT*" : "A*") << std::endl;
        resetSimulation();
        glutPostRedisplay();
        break;
    }
}

void mouse(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON)
    {
        if (state == GLUT_DOWN)
        {
            mouseLeftDown = true;
            mouseX = x;
            mouseY = y;
        }
        else if (state == GLUT_UP)
        {
            mouseLeftDown = false;
        }
    }

    if (button == 3 && state == GLUT_DOWN)
    {
        cameraDistance -= 0.5f;
        if (cameraDistance < 1.0f)
        {
            cameraDistance = 1.0f;
        }

        glutPostRedisplay();
    }
    if (button == 4 && state == GLUT_DOWN)
    {
        cameraDistance += 0.5f;
        glutPostRedisplay();
    }
}

void motion(int x, int y)
{
    if (mouseLeftDown)
    {
        int dx = x - mouseX;
        int dy = y - mouseY;

        cameraAngleY += dx * 0.5f;
        cameraAngleX += dy * 0.5f;

        if (cameraAngleX > 89.0f)
            cameraAngleX = 89.0f;
        if (cameraAngleX < -89.0f)
            cameraAngleX = -89.0f;

        mouseX = x;
        mouseY = y;

        glutPostRedisplay();
    }
}

void initSimulation()
{
    if (!g_headless)
    {
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
        buildCubeList();
    }

    simState.reset();

    for (int i = 0; i < simState.targgetParticleCount; i++)
    {
        simState.particles.push_back(Particle(0.0f, 0.0f, 0.0f));
    }

#ifdef USE_CUDA
    cuda_initPhysics(simState.targgetParticleCount);
#endif

    std::cout << "Particle System with Dynamic Pathfinding" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Particles: " << simState.targgetParticleCount << std::endl;
    std::cout << "Box size: " << boxsize << "x" << boxsize << "x" << boxsize << std::endl;
    std::cout << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  1       - Scenario: Particles Only" << std::endl;
    std::cout << "  2       - Scenario: Static Obstacles" << std::endl;
    std::cout << "  3       - Scenario: Dynamic Obstacles" << std::endl;
    std::cout << "  P       - Toggle algorithm (A* / RRT*)" << std::endl;
    std::cout << "  R       - Reset simulation" << std::endl;
    std::cout << "  W/S/A/D - Rotate camera" << std::endl;
    std::cout << "  +/-     - Zoom in/out" << std::endl;
    std::cout << "  Space   - Agitate particles" << std::endl;
    std::cout << "  Q/ESC   - Quit" << std::endl;
    std::cout << std::endl;
    std::cout << "Results CSV : " << CSV_FILE << std::endl;
    std::cout << "Metrics per run: time-to-goal, replans, plan-time (avg/min/max)," << std::endl;
    std::cout << "                 path-length, path-divergence, failed-replans" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "--headless")
            g_headless = true;
    }

    int posIdx = 0;
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "--headless")
            continue;
        posIdx++;
        if (posIdx == 1) { int n = std::atoi(argv[i]); if (n > 0) simState.targgetParticleCount = n; }
        else if (posIdx == 2) { int s = std::atoi(argv[i]); if (s >= 1 && s <= 3) simState.currentScenario = static_cast<Scenario>(s - 1); }
        else if (posIdx == 3) { simState.useRRTStar = (std::atoi(argv[i]) == 1); }
        else if (posIdx == 4) { int r = std::atoi(argv[i]); if (r > 0) g_maxRuns = r; }
    }

    initScaleFromParticleCount(simState.targgetParticleCount);

    if (g_headless)
    {
        initSimulation();
        runHeadless();
        return 0;
    }

    cameraDistance = boxsize * 2.2f;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("Particle System with Dynamic Pathfinding");

    glewInit();
    GPUinstance = initGPU();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    glClearColor(0.15f, 0.15f, 0.15f, 1.0f);

    setupLighting();

    initSimulation();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    glutMainLoop();

    return 0;
}
