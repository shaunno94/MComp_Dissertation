#pragma once
#include <inttypes.h>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm\gtc\quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

enum BUFFERS
{
	VERTEX_BUFFER, INDEX_BUFFER, MAX_BUFFER
};

#define SHADER_DIR "..\\..\\Shaders\\"

#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))

#define THREADED 1
#define CUDA 1

#define PI acos(-1.0)