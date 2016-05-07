#version 430

layout(std430, binding=6) buffer MatrixData
{
	mat4 mMatrix[];
};

uniform mat4 modelMatrix;
uniform mat4 VPMatrix;
uniform int ID;

in vec3 position;

void main()
{
	mat4 MVP;
	if (ID < 0)
		MVP = VPMatrix * modelMatrix;
	else
		MVP = VPMatrix * mMatrix[ID];

	gl_Position = MVP * vec4(position, 1.0);
}