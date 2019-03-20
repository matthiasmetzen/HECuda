#pragma once

#include "stdafx.h"
#include <iostream>
#include "device_launch_parameters.h"
#include <driver_types.h>
#include "debugCallback.h"
#include <cublasLt.h>

//#include "Kernel.cuh"

typedef struct HeightAndGradient {
	float height;
	float2 gradient;
} heightAndGradient_t;

namespace cudaHE {
	void debugInUnity(std::string message);
	void registerDebugCallback(DebugCallback callback);

	void generate(int mapSize, int numOctaves, float persistence, float lacunarity, float initialScale, bool randomizeSeed = true, int seed = 0);
	void copyMap(float* buffer, int mapSize);
	void freeMap();
	void createAndCopy(float *buffer, int mapSize, int numOctaves, float persistence, float lacunarity, float initialScale, bool randomizeSeed, int seed);

	void generateMesh(void* vertBuffer, void* triBuffer, int mapSize, float scale, float elevationScale);

	void erode(int mapSize, int n, int seed,
		int erosionRadius, float inertia, float sedimentCapacityFactor, float minSedimentCapacity,
		float erodeSpeed, float depositSpeed, float evaporateSpeed, float gravity, int maxDropletLifetime, float initialSpeed, float initialWater);

	float* getMapPointer();
}
__global__ void cudaGenerateMap(float *map, int mapSize, float initialScale, int numOctaves, float persistence, float lacunarity, int seed);
__global__ void cudaNormalize(float *map, int mapSize, float min, float diff);
__global__ void cudaGenMesh(float* vert, int* tri, float* map, int mapSize, float scale, float elevationScale);

__global__ void cudaErode(float *mapPtr, float *randData, int mapSize, int nDroplets, int erosionRadius, float inertia, float sedimentCapacityFactor, float minSedimentCapacity,
	float erodeSpeed, float depositSpeed, float evaporateSpeed, float gravity, int maxDropletLifetime, float speed, float water);
__global__ void cudaErosionBrush(int2 center, float* mapPointer, int mapSize, int radius, float amountToErode, float *sediment);

__device__ HeightAndGradient CalculateHeightAndGradient(float* mapPtr, int mapSize, float2 pos);

__device__ float3 operator*(float3 a, float b);
__device__ float3 operator+(float3 a, float3 b);
__device__ float2 operator*(float2 a, float b);
__device__ float2 operator/(float2 a, float b);
__device__ float2 operator-(float2 a, float2 b);
__device__ float2 operator-(float2 a, int2 b);
__device__ int2 operator-(int2 a, int b);
__device__ int2 operator+(int2 a, int2 b);
__device__ int2 operator-(int2 a, int2 b);
__device__ float2 operator+(float2 a, float2 b);
__device__ float3 operator*(float3 a, float b);
__device__ float3 operator+(float3 a, float3 b);
__device__ float2 normalizef2(float2 a);
__device__ float euclidNorm(float2 *a);
__device__ float euclidNorm(int2 *a);
__device__ void threadBlockDeviceSynchronize();

int cBlocks(int n);
int cThreads(int n);
dim3 cBlocks2D(int n);
dim3 cThreads2D(int n);
