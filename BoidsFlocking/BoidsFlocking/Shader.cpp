#include "Shader.h"

Shader::Shader()
{ 
	shaderProgram = 0; 
	modelMatrixLoc = 0;
	VPMatrixLoc = 0;
}

Shader::Shader(std::string vertex_file, std::string frag_file, std::string geo_file, std::string tcs_file, std::string tes_file)
{
	readFile(vertex_file, VERTEX_SHADER_BUFFER);
	readFile(frag_file, FRAG_SHADER_BUFFER);

	if (!geo_file.empty())
	{
		readFile(geo_file, GEO_SHADER_BUFFER);
	}
	if (!tcs_file.empty() && !tes_file.empty())
	{
		readFile(tcs_file, TCS_SHADER_BUFFER);
		readFile(tes_file, TES_SHADER_BUFFER);
	}
	init_shader();
	SetDefaultAttributes();
}

Shader::~Shader()
{
	//Destroy shader
	glUseProgram(0);
	glDetachShader(shaderProgram, *VERTEX_SHADER_BUFFER.c_str());
	glDetachShader(shaderProgram, *FRAG_SHADER_BUFFER.c_str());
	glDeleteShader(*VERTEX_SHADER_BUFFER.c_str());
	glDeleteShader(*FRAG_SHADER_BUFFER.c_str());
	glDeleteProgram(shaderProgram);
}

void Shader::init_shader()
{
	GLint shaderTest = 0;
	GLchar log[1024] = { 0 };
	shaderProgram = glCreateProgram();
	if (!shaderProgram)
	{
		std::cerr << "Error: Failed to create shader program." << std::endl;
		system("pause");
		exit(1);
	}

	createShader(shaderProgram, VERTEX_SHADER_BUFFER.c_str(), GL_VERTEX_SHADER);
	createShader(shaderProgram, FRAG_SHADER_BUFFER.c_str(), GL_FRAGMENT_SHADER);

	if (!GEO_SHADER_BUFFER.empty())
	{
		createShader(shaderProgram, GEO_SHADER_BUFFER.c_str(), GL_GEOMETRY_SHADER);
	}
	if (!TCS_SHADER_BUFFER.empty() && !TES_SHADER_BUFFER.empty())
	{
		createShader(shaderProgram, TCS_SHADER_BUFFER.c_str(), GL_TESS_CONTROL_SHADER);
		createShader(shaderProgram, TES_SHADER_BUFFER.c_str(), GL_TESS_EVALUATION_SHADER);
	}

	glLinkProgram(shaderProgram);
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &shaderTest);
	if (!shaderTest)
	{
		glGetProgramInfoLog(shaderProgram, sizeof(log), NULL, log);
		std::cerr << "Failed to link shader program:\n" << log << std::endl;
		system("pause");
		exit(1);
	}

	shaderTest = 0;
	glValidateProgram(shaderProgram);
	glGetProgramiv(shaderProgram, GL_VALIDATE_STATUS, &shaderTest);
	if (!shaderTest)
	{
		glGetProgramInfoLog(shaderProgram, sizeof(log), NULL, log);
		std::cerr << "Failed to validate shader program:\n" << log << std::endl;
		system("pause");
		exit(1);
	}
	glUseProgram(shaderProgram);
}

void Shader::createShader(GLuint program, const char *shader, GLenum type)
{
	GLuint shaderObj = glCreateShader(type);
	if (!shaderObj)
	{
		std::cerr << "Error: Failed to create shader object:\n" << type << std::endl;
		system("pause");
		exit(1);
	}

	const GLchar *shaderPointer = shader;
	GLint length[] = { strlen(shader) };
	glShaderSource(shaderObj, 1, &shaderPointer, length);
	glCompileShader(shaderObj);

	GLint shaderTest = 0;
	glGetShaderiv(shaderObj, GL_COMPILE_STATUS, &shaderTest);
	if (!shaderTest)
	{
		GLchar log[1024];
		glGetShaderInfoLog(shaderObj, sizeof(log), NULL, log);
		std::cerr << "Failed to compile shader:\n" << log << std::endl;
		system("pause");
		exit(1);
	}
	glAttachShader(program, shaderObj);
}

void Shader::readFile(const std::string &file, std::string& buf)
{
	std::ifstream ifs(file);
	if (!ifs)
	{
		std::cerr << "Cannot find file: " << file.c_str() << std::endl;
		system("pause");
		exit(1);
	}
	//Obtain file size & allocate memory
	ifs.seekg(0, std::ios::end);
	buf.reserve(ifs.tellg());
	ifs.seekg(0, std::ios::beg);
	//Read file
	buf.assign((std::istreambuf_iterator<char>(ifs)),
		std::istreambuf_iterator<char>());
	//Close file stream.
	ifs.close();
}

void Shader::SetDefaultAttributes()
{
	modelMatrixLoc = glGetUniformLocation(shaderProgram, "modelMatrix");
	VPMatrixLoc = glGetUniformLocation(shaderProgram, "VPMatrix");
	glBindAttribLocation(shaderProgram, VERTEX_BUFFER, "position");
	//glBindAttribLocation(shaderProgram, COLOUR_BUFFER, "colour");
	//glBindAttribLocation(shaderProgram, NORMAL_BUFFER, "normal");
	//glBindAttribLocation(shaderProgram, TANGENT_BUFFER, "tangent");
	//glBindAttribLocation(shaderProgram, TEXTURE_BUFFER, "texCoord");
}