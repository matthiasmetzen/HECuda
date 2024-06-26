// HECuda2.cpp: Definiert die exportierten Funktionen für die DLL-Anwendung.
//
#include "stdafx.h"
#include "HECuda2.cuh"
#include "Functions.cuh"

void generate(int mapSize, int numOctaves, float persistence, float lacunarity, float initialScale, bool randomizeSeed, int seed) {
	cudaHE::generate(mapSize, numOctaves, persistence, lacunarity, initialScale, randomizeSeed, seed);
}

void copyMap(float* buffer, int mapSize) {
	cudaHE::copyMap(buffer, mapSize);
}

void freeMap() {
	cudaHE::freeMap();
}

void createAndCopy(float *buffer,int mapSize, int numOctaves, float persistence, float lacunarity, float initialScale, bool randomizeSeed, int seed) {
	cudaHE::createAndCopy(buffer, mapSize, numOctaves, persistence, lacunarity, initialScale, randomizeSeed, seed);
}

float* getMapPointer() {
	return cudaHE::getMapPointer();
}

void generateMesh(void* vertBuffer, void* triBuffer, int mapSize, float scale, float elevationScale) {
	return cudaHE::generateMesh(vertBuffer, triBuffer, mapSize, scale, elevationScale);
}

void erode(int mapSize, int n, int seed,
	int erosionRadius, float inertia, float sedimentCapacityFactor, float minSedimentCapacity,
	float erodeSpeed, float depositSpeed, float evaporateSpeed, float gravity, int maxDropletLifetime, float initialSpeed, float initialWater) {

	return cudaHE::erode(mapSize, n, seed,
		erosionRadius, inertia, sedimentCapacityFactor, minSedimentCapacity, erodeSpeed,
		depositSpeed, evaporateSpeed, gravity, maxDropletLifetime, initialSpeed, initialWater);
}

void registerDebugCallback(DebugCallback callback) {
	cudaHE::registerDebugCallback(callback);
}

void debugInUnity(std::string message) {
	cudaHE::debugInUnity(message);
}