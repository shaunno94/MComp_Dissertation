#version 450
#extension GL_ARB_shader_draw_parameters : enable
//gl_DrawIDARB
layout(std430, binding=6) buffer ModelMatrixSSBO
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