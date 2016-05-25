#pragma once
#include <inttypes.h>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm\gtc\quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

//Switch between CPU and GPU computation
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
//256, 512, 1024, 2048, 3072, 5120, 8192, 10240, 20480, 30720, 40960, 51200, 61440, 71680, 81920, 100096
//150016, 200192, 250112, 500224, 750080, 1000192, 1250048
#define THREADS_PER_BLOCK 256
#else
//200, 500, 1000, 2000, 3000, 5000, 8000, 10000, 20000
//Switch threads on 
#define THREADED 1
#endif

enum BUFFERS
{
	VERTEX_BUFFER, INDEX_BUFFER, INDIRECT_BUFFER, MAX_BUFFER
};

#define SHADER_DIR "..\\..\\Shaders\\"
//#define SHADER_DIR "..\\..\\..\\Shaders\\"

#define maximum(a,b)    (((a) > (b)) ? (a) : (b))
#define minimum(a,b)    (((a) < (b)) ? (a) : (b))
#define PI acos(-1.0)