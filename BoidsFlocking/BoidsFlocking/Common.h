#pragma once
#include <inttypes.h>

enum BUFFERS
{
	VERTEX_BUFFER, INDEX_BUFFER, MAX_BUFFER
};

#define SHADER_DIR "..\\..\\Shaders\\"

#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))