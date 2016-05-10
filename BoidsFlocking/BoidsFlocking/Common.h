#pragma once
#include <inttypes.h>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm\gtc\quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#define CUDA 1
#if CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <device_functions.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#define KERNEL 2 //0 == slow, 1 == medium, 2 == fast
//1105920
//221184
//36864
#define NUM_BOIDS 221184
#define THREADS_PER_BLOCK 256
#else
#define NUM_BOIDS 5000
#endif

enum BUFFERS
{
	VERTEX_BUFFER, INDEX_BUFFER, INDIRECT_BUFFER, MAX_BUFFER
};

#define SHADER_DIR "..\\..\\Shaders\\"
//#define SHADER_DIR "..\\..\\..\\Shaders\\"

#define maximum(a,b)    (((a) > (b)) ? (a) : (b))
#define minimum(a,b)    (((a) < (b)) ? (a) : (b))

#define THREADED 1

#define PI acos(-1.0)