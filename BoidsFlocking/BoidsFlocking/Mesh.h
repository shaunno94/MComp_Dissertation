#pragma once
#include "Common.h"

class Mesh
{
public:
	Mesh(void);
	Mesh(uint32_t numVertices, glm::vec3* vertices, glm::vec2* texCoords, glm::vec3* normals, glm::vec3* tangents, uint32_t numIndices, uint32_t* indices);
	virtual ~Mesh(void);

	void Draw();

	inline void AddChild(Mesh* m) { m_Children.push_back(m); }
	inline const std::vector<Mesh*>& GetChildren() { return m_Children; }

	//Generates a single triangle, with RGB colours
	static Mesh* GenerateTriangle();
	static Mesh* GenerateSphere(uint32_t height, uint32_t width);
	//Generates a single white quad, going from -1 to 1 on the x and z axis.
	static Mesh* GenerateQuad(glm::vec2 texCoords = glm::vec2(1.0f, 1.0f));
	static Mesh* GenerateQuadAlt();

	//inline const glm::vec3& GetColour(uint32_t index) const { return m_Colours[index]; }
	inline uint32_t GetNumVertices() { return m_NumVertices; }
	inline uint32_t GetNumIndices() { return m_NumIndices; }
	inline glm::vec3* GetVertices() { return m_Vertices; }
	inline glm::vec3* GetNormals() { return m_Normals; }
	inline glm::vec3* GetTangents() { return m_Tangents; }
	inline glm::vec2* GetTextureCoords() { return m_TextureCoords; }
	inline uint32_t* GetIndices() { return m_Indices;}

	//Generates normals for all facets. Assumes geometry type is GL_TRIANGLES...
	void	GenerateNormals();
	//Generates tangents for all facets. Assumes geometry type is GL_TRIANGLES...
	void	GenerateTangents();

protected:	
	//Buffers all VBO data into graphics memory. Required before drawing!
	void BufferData();
	//Helper function for GenerateTangents
	glm::vec3 GenerateTangent(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec2& ta, const glm::vec2& tb, const glm::vec2& tc);
	void Clean();

	//Number of vertices for this mesh
	uint32_t m_NumVertices;
	//Number of indices for this mesh
	uint32_t m_NumIndices;

	glm::vec3*	m_Vertices;
	//Pointer to vertex texture coordinate attribute data
	glm::vec2*	m_TextureCoords;
	//Pointer to vertex normals attribute data
	glm::vec3*	m_Normals;
	//Pointer to vertex tangents attribute data
	glm::vec3*	m_Tangents;
	//Pointer to vertex indices attribute data
	uint32_t*	m_Indices;
	unsigned int m_PrimitiveType;
	std::vector<Mesh*> m_Children;

	//VAO for this mesh
	unsigned int arrayObject;
	//VBOs for this mesh
	unsigned int bufferObject[MAX_BUFFER];

#if CUDA
	cudaGraphicsResource* cudaVBO;
#endif
};