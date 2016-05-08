//Author: Shaun Heald
//This class loads shaders from file, compiles and links them into a working shader program, using OpenGL.
#include <GL/glew.h>
#include <iostream>
#include <string>
#include <fstream> 
#include "Common.h"

class Shader
{
public:
	Shader();
	Shader(std::string vertex_file, std::string frag_file, std::string geo_file = "", std::string tcs_file = "", std::string tes_file = "");
	~Shader();
	inline GLuint GetShaderProgram() const { return shaderProgram; }
	inline GLuint GetModelMatrixLoc() const { return modelMatrixLoc; }
	inline GLuint GetVPMatrixLoc() const { return VPMatrixLoc; }
	inline GLuint GetIDLoc() const { return IDLoc; }

private:
	void init_shader();
	void SetDefaultAttributes();
	void createShader(GLuint program, const char *shader, GLenum type);
	void readFile(const std::string &file, std::string& buf);

	GLuint shaderProgram;
	GLuint modelMatrixLoc;
	GLuint VPMatrixLoc;
	GLuint IDLoc;

	std::string VERTEX_SHADER_BUFFER;
	std::string FRAG_SHADER_BUFFER;
	std::string TCS_SHADER_BUFFER;
	std::string TES_SHADER_BUFFER;
	std::string GEO_SHADER_BUFFER;
};