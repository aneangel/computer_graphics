#include "gpu_renderer.h"

#include <GL/glew.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

bool GPUinstance = false;

static GLuint g_prog = 0;
static GLuint g_sphereVAO = 0;
static GLuint g_sphereVBO = 0;
static GLuint g_sphereEBO = 0;
static GLuint g_instVBO = 0;
static int g_sphereIdxCnt = 0;
static int g_instCap = 0;
static GLint g_uMVP = -1;
static GLint g_uMV = -1;
static GLint g_uNormalMat = -1;

static const char *VERT_SRC = R"GLSL(
#version 330 core

layout(location = 0) in vec3  aPos;
layout(location = 1) in vec3  aNormal;
layout(location = 2) in vec3  aInstPos;
layout(location = 3) in float aType;

uniform mat4 uMVP;
uniform mat4 uMV;
uniform mat3 uNormalMat;

out vec3  vNormal;
out vec3  vFragPos;
out float vType;

void main()
{
    vec3 worldPos = aPos + aInstPos;
    gl_Position   = uMVP * vec4(worldPos, 1.0);
    vFragPos      = vec3(uMV * vec4(worldPos, 1.0));
    vNormal       = normalize(uNormalMat * aNormal);
    vType         = aType;
}
)GLSL";

static const char *FRAG_SRC = R"GLSL(
#version 330 core

in vec3  vNormal;
in vec3  vFragPos;
in float vType;

out vec4 FragColor;

const vec3 L0_POS  = vec3(0.0,  10.0, 0.0);
const vec3 L0_AMB  = vec3(0.15, 0.15, 0.15);
const vec3 L0_DIFF = vec3(0.90, 0.90, 0.85);
const vec3 L0_SPEC = vec3(1.0,  1.0,  1.0);

const vec3 L1_POS  = vec3(0.0, -10.0, 0.0);
const vec3 L1_AMB  = vec3(0.05, 0.05, 0.10);
const vec3 L1_DIFF = vec3(0.20, 0.20, 0.35);

const vec3 GLOBAL_AMB = vec3(0.20, 0.20, 0.25);

void main()
{
    vec3  matDiff;
    float shininess;
    if (vType < 0.5) {
        matDiff   = vec3(0.88, 0.94, 1.0);
        shininess = 80.0;
    } else if (vType < 1.5) {
        matDiff   = vec3(0.0, 1.5, 0.3);
        shininess = 60.0;
    } else {
        matDiff   = vec3(1.0, 0.0, 0.0);
        shininess = 60.0;
    }
    vec3 matAmb  = matDiff * 0.2;
    vec3 matSpec = vec3(1.0);

    vec3 N = normalize(vNormal);
    vec3 V = normalize(-vFragPos);

    vec3  L0    = normalize(L0_POS - vFragPos);
    float diff0 = max(dot(N, L0), 0.0);
    vec3  R0    = reflect(-L0, N);
    float spec0 = pow(max(dot(V, R0), 0.0), shininess);

    vec3  L1    = normalize(L1_POS - vFragPos);
    float diff1 = max(dot(N, L1), 0.0);

    vec3 color =
        GLOBAL_AMB * matAmb
        + L0_AMB * matAmb + diff0 * L0_DIFF * matDiff + spec0 * L0_SPEC * matSpec
        + L1_AMB * matAmb + diff1 * L1_DIFF * matDiff;

    FragColor = vec4(color, 1.0);
}
)GLSL";

static GLuint compileShader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "[GPU] Shader compile error:\n"
                  << log << std::endl;
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint buildPrgm(const char *vsrc, const char *fsrc)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsrc);
    if (!vs || !fs)
    {
        glDeleteShader(vs);
        glDeleteShader(fs);
        return 0;
    }

    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetProgramInfoLog(p, 512, nullptr, log);
        std::cerr << "[GPU] Program link error:\n"
                  << log << std::endl;
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

static void buildSphere(int stacks, int slices, float radius,
                        std::vector<float> &verts,
                        std::vector<unsigned int> &idx)
{
    verts.clear();
    idx.clear();
    for (int i = 0; i <= stacks; ++i)
    {
        float phi = static_cast<float>(M_PI) * i / stacks;
        float sp = sinf(phi), cp = cosf(phi);
        for (int j = 0; j <= slices; ++j)
        {
            float theta = 2.0f * static_cast<float>(M_PI) * j / slices;
            float st = sinf(theta), ct = cosf(theta);
            float nx = sp * st, ny = cp, nz = sp * ct;
            verts.push_back(nx * radius);
            verts.push_back(ny * radius);
            verts.push_back(nz * radius);
            verts.push_back(nx);
            verts.push_back(ny);
            verts.push_back(nz);
        }
    }
    for (int i = 0; i < stacks; ++i)
    {
        for (int j = 0; j < slices; ++j)
        {
            unsigned int a = i * (slices + 1) + j;
            unsigned int b = (i + 1) * (slices + 1) + j;
            idx.push_back(a);
            idx.push_back(b);
            idx.push_back(a + 1);
            idx.push_back(b);
            idx.push_back(b + 1);
            idx.push_back(a + 1);
        }
    }
}

bool initGPU()
{
    const char *renderer = (const char *)glGetString(GL_RENDERER);
    const char *version = (const char *)glGetString(GL_VERSION);
    std::cout << "[GPU] Renderer : " << (renderer ? renderer : "unknown") << std::endl;
    std::cout << "[GPU] GL Version: " << (version ? version : "unknown") << std::endl;

    if (renderer)
    {
        std::string r(renderer);
        bool isNvidia = (r.find("NVIDIA") != std::string::npos ||
                         r.find("GeForce") != std::string::npos ||
                         r.find("Quadro") != std::string::npos ||
                         r.find("RTX") != std::string::npos);
        if (!isNvidia)
            std::cout << "[GPU] WARNING: Not running on NVIDIA GPU.\n"
                      << "[GPU]   Run './plannersim' (the generated wrapper) or rebuild with 'make'."
                      << std::endl;
    }

    GLint major = 0, minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    if (major < 3 || (major == 3 && minor < 3))
    {
        std::cout << "[GPU] OpenGL 3.3 not available (got " << major << "." << minor
                  << "); falling back to CPU path." << std::endl;
        return false;
    }

    g_prog = buildPrgm(VERT_SRC, FRAG_SRC);
    if (!g_prog)
    {
        std::cout << "[GPU] Shader compilation failed; falling back to CPU path." << std::endl;
        return false;
    }
    g_uMVP = glGetUniformLocation(g_prog, "uMVP");
    g_uMV = glGetUniformLocation(g_prog, "uMV");
    g_uNormalMat = glGetUniformLocation(g_prog, "uNormalMat");

    std::vector<float> sphereVerts;
    std::vector<unsigned int> sphereIdx;
    buildSphere(12, 16, particleRad, sphereVerts, sphereIdx);
    g_sphereIdxCnt = static_cast<int>(sphereIdx.size());

    const int MAX_INST = 2048;
    g_instCap = MAX_INST;

    glGenVertexArrays(1, &g_sphereVAO);
    glBindVertexArray(g_sphereVAO);

    glGenBuffers(1, &g_sphereVBO);
    glBindBuffer(GL_ARRAY_BUFFER, g_sphereVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(sphereVerts.size() * sizeof(float)),
                 sphereVerts.data(), GL_STATIC_DRAW);

    const GLsizei vStride = 6 * sizeof(float);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vStride, (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vStride, (void *)(3 * sizeof(float)));

    glGenBuffers(1, &g_sphereEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(sphereIdx.size() * sizeof(unsigned int)),
                 sphereIdx.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &g_instVBO);
    glBindBuffer(GL_ARRAY_BUFFER, g_instVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(MAX_INST * 4 * sizeof(float)),
                 nullptr, GL_STREAM_DRAW);

    const GLsizei iStride = 4 * sizeof(float);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, iStride, (void *)0);
    glVertexAttribDivisor(2, 1);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, iStride, (void *)(3 * sizeof(float)));
    glVertexAttribDivisor(3, 1);

    glBindVertexArray(0);

    std::cout << "[GPU] Instanced rendering active: "
              << (sphereVerts.size() / 6) << " verts/sphere, "
              << g_sphereIdxCnt << " indices — all particles in one draw call."
              << std::endl;
    return true;
}

void drawGPU(const std::vector<Particle> &particles, int robotIdx, int goalIdx)
{

    std::vector<float> instData;
    instData.reserve(particles.size() * 4);

    for (int i = 0; i < static_cast<int>(particles.size()); ++i)
    {
        const Particle &p = particles[i];
        if (!p.active)
            continue;
        float type = (i == robotIdx) ? 1.0f : (i == goalIdx) ? 2.0f
                                                             : 0.0f;
        instData.push_back(p.x);
        instData.push_back(p.y);
        instData.push_back(p.z);
        instData.push_back(type);
    }

    int instCount = static_cast<int>(instData.size() / 4);
    if (instCount == 0)
        return;

    glBindBuffer(GL_ARRAY_BUFFER, g_instVBO);
    if (instCount > g_instCap)
    {
        g_instCap = instCount * 2;
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(g_instCap * 4 * sizeof(float)),
                     nullptr, GL_STREAM_DRAW);
    }
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    static_cast<GLsizeiptr>(instData.size() * sizeof(float)),
                    instData.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    float mv[16], proj[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, mv);
    glGetFloatv(GL_PROJECTION_MATRIX, proj);

    float mvp[16] = {};
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            for (int k = 0; k < 4; ++k)
                mvp[col * 4 + row] += proj[k * 4 + row] * mv[col * 4 + k];

    float nm[9] = {
        mv[0], mv[1], mv[2],
        mv[4], mv[5], mv[6],
        mv[8], mv[9], mv[10]};

    glUseProgram(g_prog);
    glUniformMatrix4fv(g_uMVP, 1, GL_FALSE, mvp);
    glUniformMatrix4fv(g_uMV, 1, GL_FALSE, mv);
    glUniformMatrix3fv(g_uNormalMat, 1, GL_FALSE, nm);

    glBindVertexArray(g_sphereVAO);
    glDrawElementsInstanced(GL_TRIANGLES, g_sphereIdxCnt, GL_UNSIGNED_INT,
                            nullptr, instCount);
    glBindVertexArray(0);
    glUseProgram(0);
}

void cleanupGPU()
{
    if (g_sphereVAO)
    {
        glDeleteVertexArrays(1, &g_sphereVAO);
        g_sphereVAO = 0;
    }
    if (g_sphereVBO)
    {
        glDeleteBuffers(1, &g_sphereVBO);
        g_sphereVBO = 0;
    }
    if (g_sphereEBO)
    {
        glDeleteBuffers(1, &g_sphereEBO);
        g_sphereEBO = 0;
    }
    if (g_instVBO)
    {
        glDeleteBuffers(1, &g_instVBO);
        g_instVBO = 0;
    }
    if (g_prog)
    {
        glDeleteProgram(g_prog);
        g_prog = 0;
    }
}
