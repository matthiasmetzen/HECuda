#pragma once

#include <math.h>
#include <limits>
#include <random>

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include "device_atomic_functions.h"
#include "math_constants.h"

#include "Functions.cuh"
#include "cuda-noise/include/cuda_noise.cuh"


#define CUDA_CALL( call ) do {              \
	cudaError_t result = call;              \
	if ( cudaSuccess != result ) {			\
		std::string s = std::string("CUDA error" + (std::string(" in " __FILE__) + std::string(__LINE__ + ": ") + std::string(cudaGetErrorString(result) + std::string("(" #call ")")))); \
		std::cerr << s.c_str() << std::endl;		\
		debugInUnity(s.c_str());			\
	} \
} while (0)
#define CURAND_CALL( call ) do {              \
	curandStatus_t result = call;              \
	if ( CURAND_STATUS_SUCCESS != result ) {			\
		std::string s = ("cuRAND error", result, " in ", __FILE__, __LINE__, ": "/*, cudaGetErrorString(result)*/, "(", #call, ")"); \
		std::cerr << s.c_str() << std::endl,		\
		debugInUnity(s.c_str());			\
	} \
} while (0)
#define CUBLAS_CALL( call ) do {              \
	cublasStatus_t result = call;              \
	if ( CUBLAS_STATUS_SUCCESS != result ) {			\
		std::string s = ("cuBLAS error", result, " in ", __FILE__, __LINE__, ": "/*, cudaGetErrorString(result)*/, "(", #call, ")"); \
		std::cerr << s.c_str() << std::endl,		\
		debugInUnity(s.c_str());			\
	} \
} while (0)

//return EXIT_FAILURE;


#define DEFAULT_THREADS 1024
#define SEED_MIN 0
#define SEED_MAX 10000

int cBlocks(int n) {
	int threads = cThreads(n);
	return (n > 0) ? (n / threads  + (n % threads > 0)) : 1;
}

dim3 cBlocks2D(int n) {
	dim3 threads = cThreads2D(n);
	int blocks = (n > 0) ? (n / (threads.x*threads.y) + (n % (threads.x*threads.y) > 0)) : 1;
	//if ((n & (n - 1)) == 0) {
	//	throw std::invalid_argument("\'threads\' needs to be a power of 2");
	//}
	int rt = ceil(sqrt(blocks));
	return dim3(rt, rt);
}

int cThreads(int n) {
	return (n > DEFAULT_THREADS) ? DEFAULT_THREADS : (n>0 ? n : 1);
}

dim3 cThreads2D(int n) {
	int threads = (n > DEFAULT_THREADS) ? DEFAULT_THREADS : (n > 0 ? n : 1);
	//if ((threads & (threads - 1)) == 0) {
	//	throw std::invalid_argument("\'threads\' needs to be a power of 2");
	//}
	int rt = ceil(sqrt(threads));
	return dim3(rt, rt);
}

namespace cudaHE {
	float* mapPointer;
	bool isAllocated = false;
	DebugCallback gDebugCallback;

	inline int blocks(int size, int threads) {
		return size / threads + (int)((size % threads) > 0);
	}

	void generate(int mapSize, int numOctaves, float persistence, float lacunarity, float initialScale, bool randomizeSeed, int seed) {
		int maxIndex;
		int minIndex;
		float minVal;
		float maxVal;

		if (isAllocated) {
			CUDA_CALL( cudaFree(mapPointer) ); 
			isAllocated = false;
		}
		
		CUDA_CALL( cudaMalloc(&mapPointer, mapSize* mapSize * sizeof(float)) );
		isAllocated = true;

		std::random_device rd;
		std::uniform_int_distribution<int> distribution(SEED_MIN, SEED_MAX);
		std::default_random_engine rng(rd());
		seed = (randomizeSeed || seed < SEED_MIN || seed > SEED_MAX) ? distribution(rng) : seed;

		cublasHandle_t cbhandle;

		CUBLAS_CALL( cublasCreate_v2(&cbhandle) );

		int pSize = mapSize * mapSize;
		int blockSize = cBlocks(pSize);
		int threadSize = cThreads(pSize);
		//dim3 blockSize = cBlocks2D(pSize);
		//dim3 threadSize = cThreads2D(pSize);
		cudaGenerateMap <<< blockSize, threadSize >>> (mapPointer, mapSize, initialScale, numOctaves, persistence, lacunarity, seed);

		CUDA_CALL( cudaGetLastError() );
		CUDA_CALL( cudaDeviceSynchronize() );

		CUBLAS_CALL( cublasIsamax(cbhandle, mapSize * mapSize, mapPointer, 1, &maxIndex) );
		CUBLAS_CALL( cublasIsamin(cbhandle, mapSize * mapSize, mapPointer, 1, &minIndex) );

		CUDA_CALL( cudaMemcpy(&minVal, &mapPointer[minIndex], sizeof(float), cudaMemcpyDeviceToHost) );
		CUDA_CALL( cudaMemcpy(&maxVal, &mapPointer[maxIndex], sizeof(float), cudaMemcpyDeviceToHost) );
		float diff = maxVal - minVal;
		
		if (diff > 0) {
			cudaNormalize << < blockSize, threadSize >> > (mapPointer, mapSize, minVal, diff);
			CUDA_CALL(cudaGetLastError() );
		}
		CUDA_CALL(cudaDeviceSynchronize());
		CUBLAS_CALL(cublasDestroy_v2(cbhandle) );
	}

	void copyMap(float *buffer, int mapSize) {
		if (isAllocated) {
			CUDA_CALL( cudaDeviceSynchronize() );
			CUDA_CALL( cudaMemcpy(buffer, mapPointer, mapSize*mapSize * sizeof(float), cudaMemcpyDeviceToHost) );
			CUDA_CALL( cudaDeviceSynchronize() );
		}
	}

	void createAndCopy(float *buffer, int mapSize, int numOctaves, float persistence, float lacunarity, float initialScale, bool randomizeSeed, int seed) {
		generate(mapSize, numOctaves, persistence, lacunarity, initialScale, randomizeSeed, seed);
		copyMap(buffer, mapSize);
	}

	void freeMap() {
		if (isAllocated) {
			cudaFree(mapPointer);
		}
		isAllocated = false;
	}

	float *getMapPointer() {
		return mapPointer;
	}

	void generateMesh(void* vertBuffer, void* triBuffer, int mapSize, float scale, float elevationScale) {
		float* vert;
		int* tri;

		int vertSize = mapSize * mapSize * 3;
		int triSize = (mapSize - 1) * (mapSize - 1) * 6;

		CUDA_CALL( cudaMalloc(&vert, vertSize * sizeof(float)) );
		CUDA_CALL( cudaMalloc(&tri, triSize*sizeof(INT32)) );

		int pSize = mapSize*mapSize;

		int blockSize = cBlocks(pSize);
		int threadSize = cThreads(pSize);
		//dim3 blockSize = cBlocks2D(pSize);
		//dim3 threadSize = cThreads2D(pSize);
		cudaGenMesh << < blockSize, threadSize >> > (vert, tri, mapPointer, mapSize, scale, elevationScale);
		CUDA_CALL( cudaGetLastError() );
		
		
		CUDA_CALL( cudaDeviceSynchronize() );

		CUDA_CALL( cudaMemcpy(vertBuffer, vert, vertSize * sizeof(float), cudaMemcpyDeviceToHost) );
		CUDA_CALL( cudaMemcpy(triBuffer, tri, triSize * sizeof(INT32), cudaMemcpyDeviceToHost) );

		CUDA_CALL( cudaDeviceSynchronize() );

		CUDA_CALL( cudaFree(vert) );
		CUDA_CALL( cudaFree(tri) );
	}

	void erode(int mapSize, int n, int seed,
		int erosionRadius, float inertia, float sedimentCapacityFactor, float minSedimentCapacity,
		float erodeSpeed, float depositSpeed, float evaporateSpeed, float gravity, int maxDropletLifetime, float initialSpeed, float initialWater) {

		curandGenerator_t gen;
		float* randData;
		CUDA_CALL( cudaMalloc(&randData, 2 * n * sizeof(float)) );
		CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
		CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, seed) );
		CURAND_CALL( curandGenerateUniform(gen, randData, 2 * n) );
		CUDA_CALL( cudaDeviceSynchronize() );

		int pSize = n;

		int blockSize = cBlocks(pSize);
		int threadSize = cThreads(pSize);

		cudaErode << < blockSize, threadSize>> > (mapPointer, randData, mapSize, n, erosionRadius, inertia, sedimentCapacityFactor, minSedimentCapacity, erodeSpeed,
			depositSpeed, evaporateSpeed, gravity, maxDropletLifetime, initialSpeed, initialWater);

		CUDA_CALL(cudaGetLastError() );

		CUDA_CALL( cudaDeviceSynchronize() );
		CURAND_CALL( curandDestroyGenerator(gen) );
		CUDA_CALL( cudaFree(randData) );
	}

	void registerDebugCallback(DebugCallback callback) {
		if (callback) {
			gDebugCallback = callback;
		}
	}

	void debugInUnity(std::string message) {
		if (gDebugCallback) {
			gDebugCallback(message.c_str());
		}
	}
	
}


__global__ void cudaGenerateMap(float *map, int mapSize, float initialScale, int numOctaves, float persistence, float lacunarity, int seed) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int2 pos = make_int2(index % mapSize, index / mapSize);
	//int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	if (pos.x >= 0 && pos.x < mapSize && pos.y >= 0 && pos.y < mapSize) {
		//int index = pos.y * mapSize + pos.x;
		map[index] = cudaNoise::clampToUnsigned(cudaNoise::repeaterPerlin(make_float3(pos.x, pos.y, 0.0), initialScale/mapSize, seed, numOctaves, lacunarity, persistence));
	}
}

__global__ void cudaNormalize(float *map, int mapSize, float min, float diff) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;

	if (index >= 0 && index < mapSize * mapSize) {
		map[index] = (map[index] - min) / diff;
	}
}

__global__ void cudaGenMesh(float* vert, INT32* tri, float* map, int mapSize, float scale, float elevationScale) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index > 0 && index < mapSize*mapSize) {
		int x = index % mapSize;
		int y = (int)(index / (float)mapSize);
		int t = 6 * ((mapSize - 1) * y + x);

		float2 percent = make_float2(x / (mapSize - 1.0), y / (mapSize - 1.0));
		float3 pos = make_float3(percent.x * 2 - 1, 0, percent.y * 2 - 1) * scale;
		pos = pos + make_float3(0, 1, 0) * map[index] * elevationScale;

		vert[index * 3 + 0] = pos.x;
		vert[index * 3 + 1] = pos.y;
		vert[index * 3 + 2] = pos.z;

		// Construct triangles
		if (x != mapSize - 1 && y != mapSize - 1)
		{
			tri[t + 0] = index + mapSize;
			tri[t + 1] = index + mapSize + 1;
			tri[t + 2] = index;

			tri[t + 3] = index + mapSize + 1;
			tri[t + 4] = index + 1;
			tri[t + 5] = index;
		}
	}
}

__global__ void cudaErode(float *mapPtr, float *randData, int mapSize, int nDroplets, int erosionRadius, float inertia, float sedimentCapacityFactor, float minSedimentCapacity,
	float erodeSpeed, float depositSpeed, float evaporateSpeed, float gravity, int maxDropletLifetime, float initSpeed, float initWater) {

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < 0 || index >= nDroplets) {
		return;
	}

	float2 pos = make_float2(randData[2 * index], randData[2 * index + 1]) *mapSize;
	float2 direction = make_float2(.0f, .0f);
	int2 gridPos;
	int dropletIndex;
	float2 cellOffset;
	float sediment = .0f;
	float speed = initSpeed;
	float water = initWater;


	for (int lifetime = 0; lifetime < maxDropletLifetime; lifetime++) {
		 gridPos = make_int2(pos.x, pos.y);
		 dropletIndex = gridPos.y * mapSize + gridPos.x;
		 if (dropletIndex > mapSize*mapSize -(mapSize + 1) || gridPos.y < 0 || gridPos.x < 0 || gridPos.y >= mapSize - 1 || gridPos.x >= mapSize - 1) {
			 return;
		 }

		// Calculate droplet's offset inside the cell (0,0) = at NW node, (1,1) = at SE node
		cellOffset = pos - gridPos;

		// Calculate droplet's height and direction of flow with bilinear interpolation of surrounding heights
		heightAndGradient_t heightAndGradient = CalculateHeightAndGradient(mapPtr, mapSize, pos);

		// Update the droplet's direction and position (move position 1 unit regardless of speed)
		direction = (direction * inertia) - (heightAndGradient.gradient * (1 - inertia));

		// Normalize direction
		direction = normalizef2(direction);
		pos = pos+direction;

		// Stop simulating droplet if it's not moving or has flowed over edge of map
		if ((direction.x == 0 && direction.y == 0) || pos.x < 0 || pos.x >= mapSize - 1 || pos.y < 0 || pos.y >= mapSize - 1) {
			return;
		}

		// Find the droplet's new height and calculate the deltaHeight
		float newHeight = CalculateHeightAndGradient(mapPtr, mapSize, pos).height;
		float deltaHeight = newHeight - heightAndGradient.height;

		// Calculate the droplet's sediment capacity (higher when moving fast down a slope and contains lots of water)
		float sedimentCapacity = max(-deltaHeight * speed * water * sedimentCapacityFactor, minSedimentCapacity);

		// If carrying more sediment than capacity, or if flowing uphill:
		if (sediment > sedimentCapacity || deltaHeight > 0) {
			// If moving uphill (deltaHeight > 0) try fill up to the current height, otherwise deposit a fraction of the excess sediment
			float amountToDeposit = (deltaHeight > 0) ? min(deltaHeight, sediment) : (sediment - sedimentCapacity) * depositSpeed;
			sediment -= amountToDeposit;

			// Add the sediment to the four nodes of the current cell using bilinear interpolation
			// Deposition is not distributed over a radius (like erosion) so that it can fill small pits
			atomicAdd(&mapPtr[dropletIndex], amountToDeposit * (1 - cellOffset.x) * (1 - cellOffset.y));
			atomicAdd(&mapPtr[dropletIndex + 1], amountToDeposit * cellOffset.x * (1 - cellOffset.y));
			atomicAdd(&mapPtr[dropletIndex + mapSize], amountToDeposit * (1 - cellOffset.x) * cellOffset.y);
			atomicAdd(&mapPtr[dropletIndex + mapSize + 1], amountToDeposit * cellOffset.x * cellOffset.y);

		}
		else {
			// Erode a fraction of the droplet's current carry capacity.
			// Clamp the erosion to the change in height so that it doesn't dig a hole in the terrain behind the droplet
			float amountToErode = min((sedimentCapacity - sediment) * erodeSpeed, -deltaHeight);

			// Use erosion brush to erode from all nodes inside the droplet's erosion radius
			//int pSize = erosionRadius * erosionRadius;
			//int threadSize = pSize > DEFAULT_THREADS ? DEFAULT_THREADS : pSize;
			//int blockSize = (pSize > 0) ? (pSize / threadSize + (pSize % threadSize > 0)) : 1;

			//float *seds = (float*)malloc(erosionRadius*erosionRadius * sizeof(float));

			//gridPos = make_int2(pos.x, pos.y);
			//cudaErosionBrush << < blockSize, dim3(erosionRadius, erosionRadius)>> > (gridPos, mapPtr, mapSize, erosionRadius, amountToErode, seds);
			
			float erodeArea = CUDART_PI_F * erosionRadius * erosionRadius;
			for (int brushPoint = 0; brushPoint < 4 * erosionRadius*erosionRadius; brushPoint++)
			{
				int2 p = make_int2(brushPoint % (2*erosionRadius+1), brushPoint / (2*erosionRadius+1)) - erosionRadius;
				float dst = euclidNorm(&p);

				if (dst <= erosionRadius && gridPos.x + p.x >= 0 && gridPos.x + p.x < mapSize && gridPos.y + p.y >= 0 && gridPos.y + p.y < mapSize) {

					int nodeIndex = (gridPos.y + p.y) * mapSize + p.x + gridPos.x;

					float dstPercent = dst / erosionRadius;
					float weightedErodeAmount = amountToErode * (1 - dstPercent) / (2*erodeArea);
					float deltaSediment = (mapPtr[nodeIndex] < weightedErodeAmount) ? mapPtr[nodeIndex] : weightedErodeAmount;
					//mapPtr[nodeIndex] -= deltaSediment;
					atomicAdd(&mapPtr[nodeIndex], -deltaSediment);
					sediment += deltaSediment;

				}


			}

			/*cudaDeviceSynchronize();
			//TODO: reduce sum
			for (int i = 0; i < erosionRadius*erosionRadius; i++) {
				sediment += seds[i];
			}*/
			//free(seds);

		}
		//threadBlockDeviceSynchronize();

		// Update droplet's speed and water content
		speed = sqrtf(speed * speed + deltaHeight * gravity);
		water *= (1 - evaporateSpeed);

		__threadfence();
	}
}

__global__ void cudaErosionBrush(int2 center, float* mapPointer, int mapSize, int radius, float amountToErode, float *sediment) {
	int centerInd = center.y * mapSize + center.x;
	int2 brushPos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int index = brushPos.y*radius + brushPos.x;
	brushPos = brushPos - radius;
	int mapIndex = (center.y + brushPos.y) * mapSize + center.x + brushPos.x;

	float dst = euclidNorm(&brushPos);

	if (abs(brushPos.x) > radius || abs(brushPos.y) > radius) {
		return;
	}
	else if (center.x + brushPos.x < 0 || center.x + brushPos.x >= mapSize || center.y + brushPos.y < 0 || center.y + brushPos.y >= mapSize || dst > radius) {
		sediment[index] = 0;
		return;
	}

	float erodeArea = CUDART_PI_F * radius * radius;


	float dstPercent = dst / (float)radius;
	float weightedErodeAmount = amountToErode * (1 - dstPercent) / erodeArea;
	float deltaSediment = (mapPointer[mapIndex] < weightedErodeAmount) ? mapPointer[mapIndex] : weightedErodeAmount;

	atomicAdd(&mapPointer[mapIndex], -deltaSediment);
	__threadfence();
	//mapPointer[mapIndex] -= deltaSediment;
	sediment[index] = deltaSediment;

}


__device__ HeightAndGradient CalculateHeightAndGradient(float* mapPtr, int mapSize, float2 pos) {
	int2 gridPos = make_int2(pos.x, pos.y);
	int mapIndex = gridPos.y * mapSize + gridPos.x;
	float2 offset = pos - gridPos;

	// Calculate heights of the four nodes of the droplet's cell
	float4 nodes = make_float4(
		mapPtr[mapIndex],							//NW
		mapPtr[mapIndex + 1],						//NE
		mapPtr[mapIndex + mapSize],					//SW
		mapPtr[mapIndex + mapSize + 1]);			//SE

	// Calculate droplet's direction of flow with bilinear interpolation of height difference along the edges
	float2 gradient = make_float2(
			(nodes.y - nodes.x) * (1 - offset.y) + (nodes.w - nodes.z) * offset.y, 
			(nodes.z - nodes.x) * (1 - offset.x) + (nodes.w - nodes.y) * offset.x);

	// Calculate height with bilinear interpolation of the heights of the nodes of the cell
	float height = nodes.x * (1 - offset.x) * (1 - offset.y) + nodes.y * offset.x * (1 - offset.y) + nodes.z * (1 - offset.x) * offset.y + nodes.w * offset.x * offset.y;

	return HeightAndGradient{ height, gradient };
}


__device__ float2 operator*(float2 a, float b) {
	return make_float2(a.x * b, a.y * b);
}

__device__ float2 operator/(float2 a, float b) {
	return make_float2(a.x / b, a.y / b);
}

__device__ float2 operator-(float2 a, float2 b) {
	return make_float2(a.x - b.x, a.y - b.y);
}

__device__ float2 operator-(float2 a, int2 b) {
	return make_float2(a.x - b.x, a.y - b.y);
}

__device__ int2 operator-(int2 a, int2 b) {
	return make_int2(a.x - b.x, a.y - b.y);
}

__device__ int2 operator+(int2 a, int2 b) {
	return make_int2(a.x + b.x, a.y + b.y);
}

__device__ int2 operator-(int2 a, int b) {
	return make_int2(a.x - b, a.y - b);
}

__device__ float2 operator+(float2 a, float2 b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float3 operator*(float3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator+(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float2 normalizef2(float2 a) {
	float len = euclidNorm(&a);
	if (len != 0) {
		a.x /= len;
		a.y /= len;
	}
	return make_float2(a.x, a.y);
}

__device__ float euclidNorm(float2 *a) {
	return sqrtf(powf(a->x, 2) + powf(a->y, 2));
}

__device__ float euclidNorm(int2 *a) {
	return (float)sqrt(powf(a->x, 2) + powf(a->y, 2));
}

__device__ void threadBlockDeviceSynchronize() {
	__syncthreads();
	if (threadIdx.x == 0)
		cudaDeviceSynchronize();
	__syncthreads();
}