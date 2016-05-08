#version 450

uniform mat4 modelMatrix;
uniform mat4 VPMatrix;

in vec3 position;

void main()
{
	mat4 MVP = VPMatrix * modelMatrix;
	gl_Position = MVP * vec4(position, 1.0);
}