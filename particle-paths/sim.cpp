#include <GL/glew.h>
#include <GLUT/glut.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

float cameraAngleX = 30.0f;
float cameraAngleY = 45.0f;
float cameraDistance = 10.0f;

bool mouseLeftDown = false;
int mouseX = 0;
int mouseY = 0;

const float BOX_SIZE = 5.0f;
const float PARTICLE_RADIUS = 0.05f;
const float GRAVITY = -9.8f;
const float DAMPING = 0.5f;
const float TIME_STEP = 0.020f;

const float H = 0.15f;
const float HSQ = H * H;
const float MASS = 0.05f;
const float REST_DENSITY = 1000.0f;
const float GAS_CONSTANT = 2000.0f;
const float VISCOSITY = 0.1f;
const float POLY6 = 315.0f / (64.0f * M_PI * pow(H, 9));
const float SPIKY_GRAD = -45.0f / (M_PI * pow(H, 6));
const float VISC_LAP = 45.0f / (M_PI * pow(H, 6));

const float CELL_SIZE = H;
const int GRID_WIDTH = (int)(BOX_SIZE / CELL_SIZE) + 1;
std::vector<std::vector<int>> grid;

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float fx, fy, fz;
    float density;
    float pressure;
    bool active;
    
    Particle(float px, float py, float pz, float velx = 0.0f, float vely = 0.0f, float velz = 0.0f)
        : x(px), y(py), z(pz), vx(velx), vy(vely), vz(velz), 
          fx(0), fy(0), fz(0), density(0), pressure(0), active(false) {}
};

std::vector<Particle> particles;

const float SPAWN_X = 0.0f;
const float SPAWN_Z = 0.0f;
float spawnY = BOX_SIZE / 2.0f - 0.5f;
int particlesSpawned = 0;
int targetParticleCount = 1500;
const int PARTICLES_PER_FRAME = 15;

void setupLighting() {
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
    
    GLfloat lightPosition[] = { 0.0f, 5.0f, 0.0f, 1.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    
    GLfloat ambientLight[] = { 0.3f, 0.3f, 0.3f, 1.0f };
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
    
    GLfloat diffuseLight[] = { 0.8f, 0.8f, 0.8f, 1.0f };
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
    
    GLfloat specularLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);
    
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0f);
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05f);
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.01f);
}

void setMaterial(GLfloat r, GLfloat g, GLfloat b, GLfloat shininess) {
    GLfloat materialAmbient[] = { r * 0.2f, g * 0.2f, b * 0.2f, 1.0f };
    GLfloat materialDiffuse[] = { r, g, b, 1.0f };
    GLfloat materialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    
    glMaterialfv(GL_FRONT, GL_AMBIENT, materialAmbient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, materialDiffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, materialSpecular);
    glMaterialf(GL_FRONT, GL_SHININESS, shininess);
}

void drawWireframeCube() {
    glDisable(GL_LIGHTING);
    glColor3f(0.0f, 0.0f, 0.0f);
    glLineWidth(2.0f);
    
    float halfSize = BOX_SIZE / 2.0f;
    
    glBegin(GL_LINES);
        glVertex3f(-halfSize, -halfSize, -halfSize);
        glVertex3f( halfSize, -halfSize, -halfSize);
        glVertex3f( halfSize, -halfSize, -halfSize);
        glVertex3f( halfSize, -halfSize,  halfSize);
        glVertex3f( halfSize, -halfSize,  halfSize);
        glVertex3f(-halfSize, -halfSize,  halfSize);
        glVertex3f(-halfSize, -halfSize,  halfSize);
        glVertex3f(-halfSize, -halfSize, -halfSize);
        
        glVertex3f(-halfSize,  halfSize, -halfSize);
        glVertex3f( halfSize,  halfSize, -halfSize);
        glVertex3f( halfSize,  halfSize, -halfSize);
        glVertex3f( halfSize,  halfSize,  halfSize);
        glVertex3f( halfSize,  halfSize,  halfSize);
        glVertex3f(-halfSize,  halfSize,  halfSize);
        glVertex3f(-halfSize,  halfSize,  halfSize);
        glVertex3f(-halfSize,  halfSize, -halfSize);
        
        glVertex3f(-halfSize, -halfSize, -halfSize);
        glVertex3f(-halfSize,  halfSize, -halfSize);
        glVertex3f( halfSize, -halfSize, -halfSize);
        glVertex3f( halfSize,  halfSize, -halfSize);
        glVertex3f( halfSize, -halfSize,  halfSize);
        glVertex3f( halfSize,  halfSize,  halfSize);
        glVertex3f(-halfSize, -halfSize,  halfSize);
        glVertex3f(-halfSize,  halfSize,  halfSize);
    glEnd();
    
    glEnable(GL_LIGHTING);
}

void drawParticle(int numParticles) {
    (void)numParticles;
    
    for (size_t i = 0; i < particles.size(); i++) {
        if (!particles[i].active) continue;
        
        glPushMatrix();
        float densityRatio = particles[i].density / REST_DENSITY;
        float blue = 0.3f + 0.7f * std::min(densityRatio, 1.0f);
        setMaterial(0.2f, 0.5f, blue, 50.0f);
        glTranslatef(particles[i].x, particles[i].y, particles[i].z);
        glutSolidSphere(PARTICLE_RADIUS, 8, 8);
        glPopMatrix();
    }
}

void spawnParticles() {
    for (int i = 0; i < PARTICLES_PER_FRAME && particlesSpawned < targetParticleCount; i++) {
        if (particlesSpawned < (int)particles.size()) {
            float offsetRange = H * 0.3f;
            particles[particlesSpawned].x = SPAWN_X + ((rand() % 100) / 100.0f - 0.5f) * offsetRange;
            particles[particlesSpawned].y = spawnY;
            particles[particlesSpawned].z = SPAWN_Z + ((rand() % 100) / 100.0f - 0.5f) * offsetRange;
            
            particles[particlesSpawned].vx = ((rand() % 100) / 100.0f - 0.5f) * 0.2f;
            particles[particlesSpawned].vy = -0.1f;
            particles[particlesSpawned].vz = ((rand() % 100) / 100.0f - 0.5f) * 0.2f;
            
            particles[particlesSpawned].active = true;
            particlesSpawned++;
        }
    }
}

int gridHash(int x, int y, int z) {
    return (x * 92837111) ^ (y * 689287499) ^ (z * 283923481);
}

int getGridIndex(float pos) {
    float halfSize = BOX_SIZE / 2.0f;
    return (int)((pos + halfSize) / CELL_SIZE);
}

void buildSpatialGrid() {
    int gridSize = GRID_WIDTH * GRID_WIDTH * GRID_WIDTH;
    grid.clear();
    grid.resize(gridSize);
    
    for (size_t i = 0; i < particles.size(); i++) {
        if (!particles[i].active) continue;
        
        int gx = getGridIndex(particles[i].x);
        int gy = getGridIndex(particles[i].y);
        int gz = getGridIndex(particles[i].z);
        
        gx = std::max(0, std::min(GRID_WIDTH - 1, gx));
        gy = std::max(0, std::min(GRID_WIDTH - 1, gy));
        gz = std::max(0, std::min(GRID_WIDTH - 1, gz));
        
        int hash = gridHash(gx, gy, gz) % gridSize;
        if (hash < 0) hash += gridSize;
        grid[hash].push_back((int)i);
    }
}

void findNeighbors(size_t particleIdx, std::vector<int>& neighbors) {
    neighbors.clear();
    Particle& p = particles[particleIdx];
    
    int gx = getGridIndex(p.x);
    int gy = getGridIndex(p.y);
    int gz = getGridIndex(p.z);
    
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int cx = gx + dx;
                int cy = gy + dy;
                int cz = gz + dz;
                
                if (cx < 0 || cx >= GRID_WIDTH || 
                    cy < 0 || cy >= GRID_WIDTH || 
                    cz < 0 || cz >= GRID_WIDTH) continue;
                
                int hash = gridHash(cx, cy, cz) % grid.size();
                if (hash < 0) hash += grid.size();
                
                for (int idx : grid[hash]) {
                    if ((size_t)idx == particleIdx) continue;
                    
                    Particle& neighbor = particles[idx];
                    float dx = p.x - neighbor.x;
                    float dy = p.y - neighbor.y;
                    float dz = p.z - neighbor.z;
                    float distSq = dx*dx + dy*dy + dz*dz;
                    
                    if (distSq < HSQ) {
                        neighbors.push_back(idx);
                    }
                }
            }
        }
    }
}

void computeDensityPressure() {
    std::vector<int> neighbors;
    
    for (size_t i = 0; i < particles.size(); i++) {
        if (!particles[i].active) continue;
        
        Particle& p = particles[i];
        p.density = 0.0f;
        
        findNeighbors(i, neighbors);
        
        for (int j : neighbors) {
            Particle& neighbor = particles[j];
            float dx = p.x - neighbor.x;
            float dy = p.y - neighbor.y;
            float dz = p.z - neighbor.z;
            float distSq = dx*dx + dy*dy + dz*dz;
            
            if (distSq < HSQ) {
                p.density += MASS * POLY6 * pow(HSQ - distSq, 3);
            }
        }
        
        p.density += MASS * POLY6 * pow(HSQ, 3);
        p.pressure = GAS_CONSTANT * std::max(0.0f, p.density - REST_DENSITY);
    }
}

void computeForces() {
    std::vector<int> neighbors;
    
    for (size_t i = 0; i < particles.size(); i++) {
        if (!particles[i].active) continue;
        
        Particle& p = particles[i];
        float fpress_x = 0.0f, fpress_y = 0.0f, fpress_z = 0.0f;
        float fvisc_x = 0.0f, fvisc_y = 0.0f, fvisc_z = 0.0f;
        
        findNeighbors(i, neighbors);
        
        for (int j : neighbors) {
            Particle& neighbor = particles[j];
            float dx = p.x - neighbor.x;
            float dy = p.y - neighbor.y;
            float dz = p.z - neighbor.z;
            float dist = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist > 0.0001f && dist < H) {
                float nx = dx / dist;
                float ny = dy / dist;
                float nz = dz / dist;
                
                float pressureTerm = -MASS * (p.pressure + neighbor.pressure) / (2.0f * neighbor.density);
                float spiky = SPIKY_GRAD * pow(H - dist, 2);
                
                fpress_x += pressureTerm * spiky * nx;
                fpress_y += pressureTerm * spiky * ny;
                fpress_z += pressureTerm * spiky * nz;
                
                float viscLap = VISC_LAP * (H - dist);
                fvisc_x += VISCOSITY * MASS * (neighbor.vx - p.vx) / neighbor.density * viscLap;
                fvisc_y += VISCOSITY * MASS * (neighbor.vy - p.vy) / neighbor.density * viscLap;
                fvisc_z += VISCOSITY * MASS * (neighbor.vz - p.vz) / neighbor.density * viscLap;
            }
        }
        
        p.fx = fpress_x + fvisc_x;
        p.fy = fpress_y + fvisc_y + (MASS * GRAVITY);
        p.fz = fpress_z + fvisc_z;
    }
}

void updatePhysics() {
    float halfSize = BOX_SIZE / 2.0f;
    
    spawnParticles();
    buildSpatialGrid();
    computeDensityPressure();
    computeForces();
    
    for (auto& p : particles) {
        if (!p.active) continue;
        
        if (p.density < 0.01f) p.density = REST_DENSITY;
        
        float ax = p.fx / p.density;
        float ay = p.fy / p.density;
        float az = p.fz / p.density;
        
        p.vx += TIME_STEP * ax;
        p.vy += TIME_STEP * ay;
        p.vz += TIME_STEP * az;
        
        p.x += TIME_STEP * p.vx;
        p.y += TIME_STEP * p.vy;
        p.z += TIME_STEP * p.vz;
        
        if (p.x - PARTICLE_RADIUS < -halfSize) {
            p.x = -halfSize + PARTICLE_RADIUS;
            p.vx *= -DAMPING;
        } else if (p.x + PARTICLE_RADIUS > halfSize) {
            p.x = halfSize - PARTICLE_RADIUS;
            p.vx *= -DAMPING;
        }
        
        if (p.y - PARTICLE_RADIUS < -halfSize) {
            p.y = -halfSize + PARTICLE_RADIUS;
            p.vy *= -DAMPING;
        } else if (p.y + PARTICLE_RADIUS > halfSize) {
            p.y = halfSize - PARTICLE_RADIUS;
            p.vy *= -DAMPING;
        }
        
        if (p.z - PARTICLE_RADIUS < -halfSize) {
            p.z = -halfSize + PARTICLE_RADIUS;
            p.vz *= -DAMPING;
        } else if (p.z + PARTICLE_RADIUS > halfSize) {
            p.z = halfSize - PARTICLE_RADIUS;
            p.vz *= -DAMPING;
        }
    }
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    
    float camX = cameraDistance * sin(cameraAngleY * M_PI / 180.0f) * cos(cameraAngleX * M_PI / 180.0f);
    float camY = cameraDistance * sin(cameraAngleX * M_PI / 180.0f);
    float camZ = cameraDistance * cos(cameraAngleY * M_PI / 180.0f) * cos(cameraAngleX * M_PI / 180.0f);
    
    gluLookAt(camX, camY, camZ, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    
    drawWireframeCube();
    drawParticle(particles.size());
    
    glutSwapBuffers();
}

void reshape(int width, int height) {
    if (height == 0) height = 1;
    
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float)width / (float)height, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
}

void idle() {
    updatePhysics();
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
    (void)x;
    (void)y;
    
    switch (key) {
        case 'q':
        case 'Q':
        case 27:
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
        
        case 'r':
        case 'R':
            for (auto& p : particles) {
                p.active = false;
                p.x = SPAWN_X;
                p.y = spawnY;
                p.z = SPAWN_Z;
                p.vx = 0.0f;
                p.vy = 0.0f;
                p.vz = 0.0f;
            }
            particlesSpawned = 0;
            std::cout << "Simulation reset" << std::endl;
            break;
        
        case ' ':
            for (auto& p : particles) {
                if (!p.active) continue;
                p.vx += ((rand() % 100) / 100.0f - 0.5f) * 3.0f;
                p.vy += ((rand() % 100) / 100.0f) * 2.0f;
                p.vz += ((rand() % 100) / 100.0f - 0.5f) * 3.0f;
            }
            std::cout << "Fluid agitated" << std::endl;
            break;
    }
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            mouseLeftDown = true;
            mouseX = x;
            mouseY = y;
        } else if (state == GLUT_UP) {
            mouseLeftDown = false;
        }
    }
    else if (button == 3 && state == GLUT_DOWN) {
        cameraDistance -= 0.5f;
        if (cameraDistance < 3.0f) cameraDistance = 3.0f;
        glutPostRedisplay();
    }
    else if (button == 4 && state == GLUT_DOWN) {
        cameraDistance += 0.5f;
        if (cameraDistance > 50.0f) cameraDistance = 50.0f;
        glutPostRedisplay();
    }
}

void mouseMotion(int x, int y) {
    if (mouseLeftDown) {
        int deltaX = x - mouseX;
        int deltaY = y - mouseY;
        
        cameraAngleY += deltaX * 0.5f;
        cameraAngleX += deltaY * 0.5f;
        
        if (cameraAngleX > 89.0f) cameraAngleX = 89.0f;
        if (cameraAngleX < -89.0f) cameraAngleX = -89.0f;
        
        mouseX = x;
        mouseY = y;
        
        glutPostRedisplay();
    }
}

void initGL() {
    glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glShadeModel(GL_SMOOTH);
    setupLighting();
}

int main(int argc, char** argv) {
    std::cout << "SPH Fluid Particle Simulation" << std::endl;
    std::cout << "=============================" << std::endl;
    
    targetParticleCount = 2000;
    
    for (int i = 0; i < targetParticleCount; i++) {
        particles.push_back(Particle(SPAWN_X, spawnY, SPAWN_Z));
        particles[i].active = false;
    }
    
    std::cout << "Particles: " << targetParticleCount << std::endl;
    std::cout << "Method: SPH (Smoothed Particle Hydrodynamics)" << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  Mouse: Drag to rotate, scroll to zoom" << std::endl;
    std::cout << "  Q/ESC: Quit" << std::endl;
    std::cout << "  W/A/S/D: Rotate camera" << std::endl;
    std::cout << "  R: Reset simulation" << std::endl;
    std::cout << "  SPACE: Agitate fluid" << std::endl;
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("SPH Fluid Simulation");
    
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
        return 1;
    }
    
    initGL();
    
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseMotion);
    glutIdleFunc(idle);
    
    glutMainLoop();
    
    return 0;
}
