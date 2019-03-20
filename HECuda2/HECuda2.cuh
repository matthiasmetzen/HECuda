#pragma once

#include "stdafx.h"
#include <iostream>
#include "debugCallback.h"

#ifdef HECUDA2_EXPORTS
#define HE_API __declspec(dllexport) __cdecl
#else
#define HE_API __declspec(dllimport) __cdecl
#endif


using namespace std;


extern "C" {
	
	 void HE_API generate(int mapSize, int numOctaves, float persistence, float lacunarity, float initialScale, bool randomizeSeed = true, int seed = 0);

	 void HE_API copyMap(float* buffer, int mapSize);

	 void HE_API freeMap();

	 void HE_API createAndCopy(float *buffer, int mapSize, int numOctaves, float persistence, float lacunarity, float initialScale, bool randomizeSeed, int seed);

	 float* HE_API getMapPointer();

	 void HE_API generateMesh(void* vertBuffer, void* triBuffer, int mapSize, float scale, float elevationScale);

	 void HE_API erode(int mapSize, int n, int seed,
		 int erosionRadius, float inertia, float sedimentCapacityFactor, float minSedimentCapacity,
		 float erodeSpeed, float depositSpeed, float evaporateSpeed, float gravity, int maxDropletLifetime, float initialSpeed, float initialWater);

	 void HE_API registerDebugCallback(DebugCallback callback);

	 void HE_API debugInUnity(std::string message);
}