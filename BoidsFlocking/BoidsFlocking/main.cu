#include "Common.h"
#if CUDA
#include "OGLRenderer.h"
#include "BoidGeneratorGPU.h"
#include "Timer.h"
#include "Shader.h"
#include "Mesh.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <device_functions.h>

const uint32_t NUM_BOIDS = 8192;
const uint32_t THREADS_K1 = 32;
const uint32_t THREADS_K2 = 1024;
const uint32_t BLOCKS_K1 = (NUM_BOIDS + (THREADS_K1 - 1)) / THREADS_K1;
const uint32_t BLOCKS_K2 = (NUM_BOIDS + (THREADS_K2 - 1)) / THREADS_K2;
const dim3 THREAD_DIM_K1 = dim3(THREADS_K1, THREADS_K1);
const dim3 BLOCK_DIM_K1 = dim3(BLOCKS_K1, BLOCKS_K1);
#define MAX_DIST 90.0f

//while (tid < NUM_BOIDS)//tid += blockDim.x * gridDim.x;		//int offset = x + y * blockDim.x * gridDim.x;
//cudaMalloc((void**)&dev_a, NUM_BOIDS * sizeof(BoidGPU));//cudaMemcpy(dev_a, boidGPU->GetBoidData().data(), NUM_BOIDS * sizeof(BoidGPU), cudaMemcpyHostToDevice);

__global__ void compute_KNN(BoidGPU* boid)
{	
	int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

	if (glm::distance(boid[tid_x].m_Position, boid[tid_y].m_Position) <= MAX_DIST)
	{	
		if (boid[tid_x].lastIndex < 50)
		{
			unsigned int index = atomicAdd(&boid[tid_x].lastIndex, 1);
			boid[tid_x].neighbours[index] = &boid[tid_y];
		}
	}	
}

__device__ void CalcCohesion(BoidGPU& boid, glm::vec3& cohVec)
{
	glm::vec3 avgPos = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < 50; ++i)
	{
		avgPos += boid.neighbours[i]->m_Position;
	}

	avgPos /= 50.0f;
	cohVec = (avgPos - boid.m_Position) /* 0.001f*/;

	float mag = glm::length(cohVec);
	glm::normalize(cohVec);

	cohVec *= (0.25f * (mag * 0.001f));
	cohVec -= boid.m_Velocity;
}

__device__ void CalcSeperation(BoidGPU& boid, glm::vec3& sepVec)
{
	for (unsigned int i = 0; i < 50; ++i)
	{
		sepVec -= (boid.neighbours[i]->m_Position - boid.m_Position) / glm::distance(boid.neighbours[i]->m_Position, boid.m_Position);
	}
	sepVec /= 50.0f;
	sepVec *= 0.3f;
}

__device__ void CalcAlignment(BoidGPU& boid, glm::vec3& alignVec)
{
	glm::vec3 avgVel = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < 50; ++i)
	{
		avgVel += boid.neighbours[i]->m_Velocity;
	}
	avgVel /= 50.0f;
	alignVec = (avgVel - boid.m_Velocity) * 0.8f;
}

__global__ void updateBoids(BoidGPU* boid, float dt)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	glm::vec3 cohVec(0, 0, 0);
	glm::vec3 sepVec(0, 0, 0);
	glm::vec3 alignVec(0, 0, 0);
	CalcCohesion(boid[tid], cohVec);
	CalcSeperation(boid[tid], sepVec);
	CalcAlignment(boid[tid], alignVec);

	boid[tid].m_Velocity = cohVec + sepVec + alignVec;
	boid[tid].m_Position += (boid[tid].m_Velocity * dt);
	boid[tid].m_WorldTransform = glm::translate(glm::mat4(1.0f), boid[tid].m_Position);
	boid[tid].lastIndex = 0;
}

int main(void)
{
	OGLRenderer* renderer = OGLRenderer::Instance();
	Shader* simpleShader = new Shader(SHADER_DIR"vertex_shader.glsl", SHADER_DIR"frag_shader.glsl");
	Mesh* triMesh = Mesh::GenerateTriangle();
	cudaSetDevice(0);
	BoidGeneratorGPU boidGPU(NUM_BOIDS);
	Timer gt;

	/*compute_KNN << <BLOCK_DIM_K1, THREAD_DIM_K1 >> >(boidGPU.GetBoidData());
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

	updateBoids << <BLOCKS_K2, THREADS_K2 >> >(boidGPU.GetBoidData(), 16.0f);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

	for (unsigned int i = 0; i < NUM_BOIDS; ++i)
	{
		std::cout << boidGPU.GetBoidData()[i].lastIndex << std::endl;
	}*/

	//Main loop.
	while (renderer->ShouldClose()) //Check if the ESC key was pressed or the window was closed
	{
		gt.startTimer();
		compute_KNN << <BLOCK_DIM_K1, THREAD_DIM_K1 >> >(boidGPU.GetBoidData());
		updateBoids << <BLOCKS_K2, THREADS_K2 >> >(boidGPU.GetBoidData(), 16.0f);
		
		renderer->Render(gt.getLast());
		glfwPollEvents();
		gt.stopTimer();
	}

	OGLRenderer::Release();
	delete triMesh;
	delete simpleShader;
	cudaDeviceReset();
	return 0;
}
#endif