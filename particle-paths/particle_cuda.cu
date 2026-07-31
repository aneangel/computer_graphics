#include "particle_cuda.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                              \
    do                                                             \
    {                                                              \
        cudaError_t _err = (x);                                    \
        if (_err != cudaSuccess)                                   \
        {                                                          \
            fprintf(stderr, "CUDA error: %s  at %s:%d\n",          \
                    cudaGetErrorString(_err), __FILE__, __LINE__); \
            exit(1);                                               \
        }                                                          \
    } while (0)

static float *d_x, *d_y, *d_z;
static float *d_vx, *d_vy, *d_vz;
static float *d_fx, *d_fy, *d_fz;
static float *d_density, *d_pressure;
static int *d_active;

static int *d_cellIdx;
static int *d_sortedPid;
static int *d_cellStart;
static int *d_cellEnd;

static float *h_x, *h_y, *h_z;
static float *h_vx, *h_vy, *h_vz;
static float *h_density;
static int *h_active;

static int g_gridSize = 0;

__global__ void computeCellIdx(
    const float *__restrict__ x,
    const float *__restrict__ y,
    const float *__restrict__ z,
    const int *__restrict__ active,
    int *cellIdx, int *sortedPid,
    int N, int gw, float halfSize, float cs, int sentinelCell)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    sortedPid[i] = i;
    if (!active[i])
    {
        cellIdx[i] = sentinelCell;
        return;
    }
    int gx = (int)((x[i] + halfSize) / cs);
    int gy = (int)((y[i] + halfSize) / cs);
    int gz = (int)((z[i] + halfSize) / cs);
    gx = max(0, min(gw - 1, gx));
    gy = max(0, min(gw - 1, gy));
    gz = max(0, min(gw - 1, gz));
    cellIdx[i] = gx * gw * gw + gy * gw + gz;
}

__global__ void buildCellRanges(
    const int *__restrict__ cellIdx,
    int *cellStart, int *cellEnd,
    int N, int gridSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    int cell = cellIdx[i];
    if (cell >= gridSize)
        return;
    if (i == 0 || cellIdx[i - 1] != cell)
        cellStart[cell] = i;
    if (i == N - 1 || cellIdx[i + 1] != cell)
        cellEnd[cell] = i + 1;
}

__global__ void computeDensityPressure(
    const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
    const int *__restrict__ sortedPid,
    const int *__restrict__ cellStart, const int *__restrict__ cellEnd,
    float *density, float *pressure,
    int N, int gw, int gridSize,
    float halfSize, float cs, float HSQ,
    float MASS_c, float POLY6_c, float REST_DENSITY_c, float GAS_CONSTANT_c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    int i = sortedPid[idx];

    float px = x[i], py = y[i], pz = z[i];
    int gx = (int)((px + halfSize) / cs);
    int gy = (int)((py + halfSize) / cs);
    int gz = (int)((pz + halfSize) / cs);
    gx = max(0, min(gw - 1, gx));
    gy = max(0, min(gw - 1, gy));
    gz = max(0, min(gw - 1, gz));

    float rho = MASS_c * POLY6_c * (HSQ * HSQ * HSQ);

    for (int ddx = -1; ddx <= 1; ddx++)
    {
        int cx = gx + ddx;
        if (cx < 0 || cx >= gw)
            continue;
        for (int ddy = -1; ddy <= 1; ddy++)
        {
            int cy = gy + ddy;
            if (cy < 0 || cy >= gw)
                continue;
            for (int ddz = -1; ddz <= 1; ddz++)
            {
                int cz = gz + ddz;
                if (cz < 0 || cz >= gw)
                    continue;
                int cell = cx * gw * gw + cy * gw + cz;
                int st = cellStart[cell];
                if (st < 0)
                    continue;
                int en = cellEnd[cell];
                for (int k = st; k < en; k++)
                {
                    int j = sortedPid[k];
                    if (j == i)
                        continue;
                    float ex = px - x[j];
                    float ey = py - y[j];
                    float ez = pz - z[j];
                    float dSq = ex * ex + ey * ey + ez * ez;
                    if (dSq < HSQ)
                    {
                        float diff = HSQ - dSq;
                        rho += MASS_c * POLY6_c * (diff * diff * diff);
                    }
                }
            }
        }
    }

    density[i] = rho;
    pressure[i] = GAS_CONSTANT_c * fmaxf(0.f, rho - REST_DENSITY_c);
}

__global__ void computeForces(
    const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
    const float *__restrict__ vx, const float *__restrict__ vy, const float *__restrict__ vz,
    const float *__restrict__ density, const float *__restrict__ pressure,
    const int *__restrict__ sortedPid,
    const int *__restrict__ cellStart, const int *__restrict__ cellEnd,
    float *fx, float *fy, float *fz,
    int N, int gw, int gridSize,
    float halfSize, float cs,
    float MASS_c, float VISCOSITY_c,
    float SPIKY_GRAD_c, float VISC_LAP_c, float gravity_c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    int i = sortedPid[idx];

    float px = x[i], py = y[i], pz = z[i];
    float pvx = vx[i], pvy = vy[i], pvz = vz[i];
    float pi_p = pressure[i];

    int gx = (int)((px + halfSize) / cs);
    int gy = (int)((py + halfSize) / cs);
    int gz = (int)((pz + halfSize) / cs);
    gx = max(0, min(gw - 1, gx));
    gy = max(0, min(gw - 1, gy));
    gz = max(0, min(gw - 1, gz));

    float fpx = 0.f, fpy = 0.f, fpz = 0.f;
    float fvx = 0.f, fvy = 0.f, fvz = 0.f;

    for (int ddx = -1; ddx <= 1; ddx++)
    {
        int cx = gx + ddx;
        if (cx < 0 || cx >= gw)
            continue;
        for (int ddy = -1; ddy <= 1; ddy++)
        {
            int cy = gy + ddy;
            if (cy < 0 || cy >= gw)
                continue;
            for (int ddz = -1; ddz <= 1; ddz++)
            {
                int cz = gz + ddz;
                if (cz < 0 || cz >= gw)
                    continue;
                int cell = cx * gw * gw + cy * gw + cz;
                int st = cellStart[cell];
                if (st < 0)
                    continue;
                int en = cellEnd[cell];
                for (int k = st; k < en; k++)
                {
                    int j = sortedPid[k];
                    if (j == i)
                        continue;
                    float ex = px - x[j];
                    float ey = py - y[j];
                    float ez = pz - z[j];
                    float dist = sqrtf(ex * ex + ey * ey + ez * ez);
                    if (dist > 0.0001f && dist < cs)
                    {
                        float nx = ex / dist;
                        float ny = ey / dist;
                        float nz = ez / dist;
                        float pterm = -MASS_c * (pi_p + pressure[j]) / (2.f * density[j]);
                        float spiky = SPIKY_GRAD_c * (cs - dist) * (cs - dist);
                        fpx += pterm * spiky * nx;
                        fpy += pterm * spiky * ny;
                        fpz += pterm * spiky * nz;
                        float vl = VISC_LAP_c * (cs - dist);
                        fvx += VISCOSITY_c * MASS_c * (vx[j] - pvx) / density[j] * vl;
                        fvy += VISCOSITY_c * MASS_c * (vy[j] - pvy) / density[j] * vl;
                        fvz += VISCOSITY_c * MASS_c * (vz[j] - pvz) / density[j] * vl;
                    }
                }
            }
        }
    }

    fx[i] = fpx + fvx;
    fy[i] = fpy + fvy + MASS_c * gravity_c;
    fz[i] = fpz + fvz;
}

__global__ void integrate(
    float *x, float *y, float *z,
    float *vx, float *vy, float *vz,
    const float *__restrict__ fx,
    const float *__restrict__ fy,
    const float *__restrict__ fz,
    const float *__restrict__ density,
    const int *__restrict__ active,
    int N, float halfSize, float particleRad_c,
    float timeStep_c, float damping_c, float REST_DENSITY_c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || !active[i])
        return;

    float rho = density[i] < 0.01f ? REST_DENSITY_c : density[i];
    vx[i] += timeStep_c * fx[i] / rho;
    vy[i] += timeStep_c * fy[i] / rho;
    vz[i] += timeStep_c * fz[i] / rho;
    x[i] += timeStep_c * vx[i];
    y[i] += timeStep_c * vy[i];
    z[i] += timeStep_c * vz[i];

    float lo = -halfSize + particleRad_c;
    float hi = halfSize - particleRad_c;

    if (x[i] < lo)
    {
        x[i] = lo;
        vx[i] *= -damping_c;
    }
    else if (x[i] > hi)
    {
        x[i] = hi;
        vx[i] *= -damping_c;
    }
    if (y[i] < lo)
    {
        y[i] = lo;
        vy[i] *= -damping_c;
    }
    else if (y[i] > hi)
    {
        y[i] = hi;
        vy[i] *= -damping_c;
    }
    if (z[i] < lo)
    {
        z[i] = lo;
        vz[i] *= -damping_c;
    }
    else if (z[i] > hi)
    {
        z[i] = hi;
        vz[i] *= -damping_c;
    }
}

void cuda_initPhysics(int maxParticles)
{
    g_gridSize = gridWidth * gridWidth * gridWidth;

    size_t fsz = maxParticles * sizeof(float);
    size_t isz = maxParticles * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_x, fsz));
    CUDA_CHECK(cudaMalloc(&d_y, fsz));
    CUDA_CHECK(cudaMalloc(&d_z, fsz));
    CUDA_CHECK(cudaMalloc(&d_vx, fsz));
    CUDA_CHECK(cudaMalloc(&d_vy, fsz));
    CUDA_CHECK(cudaMalloc(&d_vz, fsz));
    CUDA_CHECK(cudaMalloc(&d_fx, fsz));
    CUDA_CHECK(cudaMalloc(&d_fy, fsz));
    CUDA_CHECK(cudaMalloc(&d_fz, fsz));
    CUDA_CHECK(cudaMalloc(&d_density, fsz));
    CUDA_CHECK(cudaMalloc(&d_pressure, fsz));
    CUDA_CHECK(cudaMalloc(&d_active, isz));
    CUDA_CHECK(cudaMalloc(&d_cellIdx, isz));
    CUDA_CHECK(cudaMalloc(&d_sortedPid, isz));
    CUDA_CHECK(cudaMalloc(&d_cellStart, (g_gridSize + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellEnd, (g_gridSize + 1) * sizeof(int)));

    CUDA_CHECK(cudaMallocHost(&h_x, fsz));
    CUDA_CHECK(cudaMallocHost(&h_y, fsz));
    CUDA_CHECK(cudaMallocHost(&h_z, fsz));
    CUDA_CHECK(cudaMallocHost(&h_vx, fsz));
    CUDA_CHECK(cudaMallocHost(&h_vy, fsz));
    CUDA_CHECK(cudaMallocHost(&h_vz, fsz));
    CUDA_CHECK(cudaMallocHost(&h_density, fsz));
    CUDA_CHECK(cudaMallocHost(&h_active, isz));

    printf("CUDA physics ready: max %d particles, grid %d^3 = %d cells\n",
           maxParticles, gridWidth, g_gridSize);
}

void cuda_cleanupPhysics()
{
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
    cudaFree(d_density);
    cudaFree(d_pressure);
    cudaFree(d_active);
    cudaFree(d_cellIdx);
    cudaFree(d_sortedPid);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);

    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFreeHost(h_z);
    cudaFreeHost(h_vx);
    cudaFreeHost(h_vy);
    cudaFreeHost(h_vz);
    cudaFreeHost(h_density);
    cudaFreeHost(h_active);
}

void cuda_updatePhysics(SimulationState &simState)
{
    spawnParticles(simState);

    int N = (int)simState.particles.size();
    if (N == 0)
        return;

    for (int i = 0; i < N; i++)
    {
        const Particle &p = simState.particles[i];
        h_x[i] = p.x;
        h_y[i] = p.y;
        h_z[i] = p.z;
        h_vx[i] = p.vx;
        h_vy[i] = p.vy;
        h_vz[i] = p.vz;
        h_active[i] = p.active ? 1 : 0;
    }

    size_t fsz = N * sizeof(float);
    size_t isz = N * sizeof(int);
    CUDA_CHECK(cudaMemcpy(d_x, h_x, fsz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, fsz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z, fsz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, fsz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, fsz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz, fsz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_active, h_active, isz, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float halfSize = boxsize / 2.f;
    int sentinelCell = g_gridSize;

    CUDA_CHECK(cudaMemset(d_cellStart, -1, (g_gridSize + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cellEnd, 0, (g_gridSize + 1) * sizeof(int)));

    computeCellIdx<<<blocks, threads>>>(
        d_x, d_y, d_z, d_active,
        d_cellIdx, d_sortedPid,
        N, gridWidth, halfSize, cellSize, sentinelCell);

    thrust::device_ptr<int> tp_key(d_cellIdx);
    thrust::device_ptr<int> tp_val(d_sortedPid);
    thrust::sort_by_key(tp_key, tp_key + N, tp_val);

    buildCellRanges<<<blocks, threads>>>(
        d_cellIdx, d_cellStart, d_cellEnd, N, g_gridSize);

    computeDensityPressure<<<blocks, threads>>>(
        d_x, d_y, d_z,
        d_sortedPid, d_cellStart, d_cellEnd,
        d_density, d_pressure,
        N, gridWidth, g_gridSize, halfSize, cellSize, HSQ,
        MASS, POLY6, REST_DENISITY, GAS_CONSTANT);

    computeForces<<<blocks, threads>>>(
        d_x, d_y, d_z, d_vx, d_vy, d_vz,
        d_density, d_pressure,
        d_sortedPid, d_cellStart, d_cellEnd,
        d_fx, d_fy, d_fz,
        N, gridWidth, g_gridSize, halfSize, cellSize,
        MASS, VISCOSITY, SPIKY_GRAD, VISC_LAP, gravity);

    integrate<<<blocks, threads>>>(
        d_x, d_y, d_z, d_vx, d_vy, d_vz,
        d_fx, d_fy, d_fz,
        d_density, d_active,
        N, halfSize, particleRad, timeStep, damping, REST_DENISITY);

    CUDA_CHECK(cudaMemcpy(h_x, d_x, fsz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, fsz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z, d_z, fsz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vx, d_vx, fsz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy, d_vy, fsz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz, d_vz, fsz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_density, d_density, fsz, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
    {
        Particle &p = simState.particles[i];
        p.x = h_x[i];
        p.y = h_y[i];
        p.z = h_z[i];
        p.vx = h_vx[i];
        p.vy = h_vy[i];
        p.vz = h_vz[i];
        p.density = h_density[i];
    }
}
