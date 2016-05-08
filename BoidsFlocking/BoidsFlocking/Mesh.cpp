#include "Mesh.h"
#include <GL/glew.h>

Mesh::Mesh(void)
{
	m_NumVertices = 0;
	m_PrimitiveType = 0;
	m_NumIndices = 0;
	m_Vertices = nullptr;
	m_TextureCoords = nullptr;
	m_Normals = nullptr;
	m_Tangents = nullptr;
	m_Indices = nullptr;
	m_Children.clear();
	arrayObject = 0;

	for (int i = 0; i < MAX_BUFFER; ++i)
		bufferObject[i] = 0;
}

Mesh::Mesh(uint32_t numVertices, glm::vec3* vertices, glm::vec2* texCoords, glm::vec3* normals, glm::vec3* tangents, uint32_t numIndices, uint32_t* indices)
{
	m_Children.clear();

	m_NumVertices = numVertices;
	m_NumIndices = numIndices;
	m_Vertices = vertices;
	m_TextureCoords = texCoords;
	m_Normals = normals;
	m_Tangents = tangents;
	m_Indices = indices;
	m_PrimitiveType = GL_TRIANGLE_STRIP;
	arrayObject = 0;

	for (unsigned int i = 0; i < MAX_BUFFER; ++i)
		bufferObject[i] = 0;

	BufferData();
}

Mesh::~Mesh(void)
{
	Clean();
	glDeleteBuffers(MAX_BUFFER, bufferObject);
}

void Mesh::Clean()
{
	if (m_Vertices)
	{
		delete[] m_Vertices;
		m_Vertices = nullptr;
	}
	if (m_Indices)
	{
		delete[] m_Indices;
		m_Indices = nullptr;
	}
	if (m_TextureCoords)
	{
		delete[] m_TextureCoords;
		m_TextureCoords = nullptr;
	}
	if (m_Tangents)
	{
		delete[] m_Tangents;
		m_Tangents = nullptr;
	}
	if (m_Normals)
	{
		delete[] m_Normals;
		m_Normals = nullptr;
	}
}

Mesh* Mesh::GenerateSphere(uint32_t height, uint32_t width)
{
	Mesh* mesh = new Mesh();
	mesh->m_NumVertices = height * width;
	mesh->m_NumIndices = (height * width) * 6;
	mesh->m_PrimitiveType = GL_TRIANGLE_STRIP;
	mesh->m_Vertices = new glm::vec3[mesh->m_NumVertices];
	mesh->m_Indices = new uint32_t[mesh->m_NumIndices];

	//Iterates through angles of phi and theta to produce sphere.
	for (uint32_t y = 0; y < height; y++)
	{
		double phi = (double(y) * PI) / (height - 1);
		for (uint32_t x = 0; x < width; x++)
		{
			double theta = (double(x) * (2 * PI)) / (width - 1);
			mesh->m_Vertices[(y * x) + x] = glm::normalize(glm::vec3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta)));
		}
	}
	for (uint32_t i = 0; i < height - 1; ++i)
	{
		mesh->m_Indices[i] = (i * width);
		for (uint32_t j = 0; j < width; ++j)
		{
			mesh->m_Indices[(i * j) + j + 1] = (i * width + j);
			mesh->m_Indices[(i * j) + j + 2] = ((i + 1) * width + j);
		}
		mesh->m_Indices[i + 2] = ((i + 1) * width + (width - 1));
	}
	return mesh;
}

Mesh* Mesh::GenerateTriangle(bool multiDraw)
{
	Mesh* mesh = new Mesh();

	mesh->m_NumVertices = 3;
	mesh->m_NumIndices = 3;
	mesh->m_PrimitiveType = GL_TRIANGLE_STRIP;

	mesh->m_Vertices = new glm::vec3[mesh->m_NumVertices];
	mesh->m_TextureCoords = new glm::vec2[mesh->m_NumVertices];
	mesh->m_Normals = new glm::vec3[mesh->m_NumVertices];
	mesh->m_Tangents = new glm::vec3[mesh->m_NumVertices];
	mesh->m_Indices = new uint32_t[mesh->m_NumVertices];

	mesh->m_Vertices[0] = glm::vec3(0.0f, 0.5f, 0.0f);
	mesh->m_Vertices[1] = glm::vec3(0.5f, -0.5f, 0.0f);
	mesh->m_Vertices[2] = glm::vec3(-0.5f, -0.5f, 0.0f);

	mesh->m_TextureCoords[0] = glm::vec2(0.5f, 0.0f);
	mesh->m_TextureCoords[1] = glm::vec2(1.0f, 1.0f);
	mesh->m_TextureCoords[2] = glm::vec2(0.0f, 1.0f);

	for (unsigned int i = 0; i < mesh->m_NumVertices; ++i) {
		mesh->m_Normals[i] = glm::vec3(0, 0, 1);
		mesh->m_Tangents[i] = glm::vec3(1, 0, 0);
		mesh->m_Indices[i] = i;
	}

	mesh->BufferData(multiDraw);
	return mesh;
}

Mesh* Mesh::GenerateQuad(glm::vec2 texCoords)
{
	Mesh* m = new Mesh();

	m->m_NumVertices = 4;
	m->m_NumIndices = 4;
	m->m_PrimitiveType = GL_TRIANGLE_STRIP;

	m->m_Vertices = new glm::vec3[m->m_NumVertices];
	m->m_Indices = new uint32_t[m->m_NumIndices];
	m->m_TextureCoords = new glm::vec2[m->m_NumVertices];
	m->m_Normals = new glm::vec3[m->m_NumVertices];
	m->m_Tangents = new glm::vec3[m->m_NumVertices];

	m->m_Vertices[0] = glm::vec3(-1.0f, 1.0f, 0.0f);
	m->m_Vertices[1] = glm::vec3(-1.0f, -1.0f, 0.0f);
	m->m_Vertices[2] = glm::vec3(1.0f, 1.0f, 0.0f);
	m->m_Vertices[3] = glm::vec3(1.0f, -1.0f, 0.0f);

	m->m_TextureCoords[0] = glm::vec2(0.0f, texCoords.y);
	m->m_TextureCoords[1] = glm::vec2(0.0f, 0.0f);
	m->m_TextureCoords[2] = glm::vec2(texCoords.x, texCoords.y);
	m->m_TextureCoords[3] = glm::vec2(texCoords.x, 0.0f);

	for (unsigned int i = 0; i < m->m_NumIndices; ++i)
	{
		m->m_Normals[i] = glm::vec3(0.0f, 0.0f, -1.0f);
		m->m_Tangents[i] = glm::vec3(1.0f, 0.0f, 0.0f);
		m->m_Indices[i] = i;
	}
	m->BufferData();
	return m;
}

Mesh* Mesh::GenerateQuadAlt()
{
	Mesh* m = new Mesh();

	m->m_NumVertices = 4;
	m->m_NumIndices = 6;
	m->m_PrimitiveType = GL_TRIANGLE_STRIP;

	m->m_Vertices = new glm::vec3[m->m_NumVertices];
	m->m_Indices = new uint32_t[m->m_NumIndices];
	m->m_TextureCoords = new glm::vec2[m->m_NumVertices];
	m->m_Normals = new glm::vec3[m->m_NumVertices];
	m->m_Tangents = new glm::vec3[m->m_NumVertices];

	m->m_Vertices[0] = glm::vec3(0.0f, 0.0f, 0.0f);
	m->m_Vertices[1] = glm::vec3(0.0f, 1.0f, 0.0f);
	m->m_Vertices[2] = glm::vec3(1.0f, 0.0f, 0.0f);
	m->m_Vertices[3] = glm::vec3(1.0f, 1.0f, 0.0f);

	m->m_TextureCoords[0] = glm::vec2(0.0f, 0.0f);
	m->m_TextureCoords[1] = glm::vec2(0.0f, 1.0f);
	m->m_TextureCoords[2] = glm::vec2(1.0f, 0.0f);
	m->m_TextureCoords[3] = glm::vec2(1.0f, 1.0f);

	for (unsigned int i = 0; i < m->m_NumIndices; ++i)
	{
		m->m_Normals[i] = glm::vec3(0.0f, 0.0f, -1.0f);
		m->m_Tangents[i] = glm::vec3(1.0f, 0.0f, 0.0f);
		m->m_Indices[i] = i;
	}
	m->BufferData();
	return m;
}

void Mesh::GenerateNormals()
{
	if (!m_Normals)
		m_Normals = new glm::vec3[m_NumVertices];
	else
		return;

	for (unsigned int i = 0; i < m_NumVertices; ++i)
		m_Normals[i] = glm::vec3(0, 0, 0);

	if (m_Indices)
	{
		for (unsigned int i = 0; i < m_NumIndices; i += 3)
		{
			int a = m_Indices[i];
			int b = m_Indices[i + 1];
			int c = m_Indices[i + 2];

			glm::vec3 normal = glm::cross(m_Vertices[b] - m_Vertices[a], m_Vertices[c] - m_Vertices[a]);

			m_Normals[a] += normal;
			m_Normals[b] += normal;
			m_Normals[c] += normal;
		}
	}
	else
	{
		//It's just a list of triangles, so generate face normals
		for (unsigned int i = 0; i < m_NumVertices; i += 3)
		{
			glm::vec3& a = m_Vertices[i];
			glm::vec3& b = m_Vertices[i + 1];
			glm::vec3& c = m_Vertices[i + 2];

			glm::vec3 normal = glm::cross(a - b, a - c);

			m_Normals[i] = normal;
			m_Normals[i + 1] = normal;
			m_Normals[i + 2] = normal;
		}
	}

	for (unsigned int i = 0; i < m_NumVertices; ++i)
		glm::normalize(m_Normals[i]);
}

void Mesh::GenerateTangents()
{
	//Extra! stops rare occurrence of this function being called
	//on a mesh without tex coords, which would break quite badly!
	if (!m_TextureCoords)
		return;

	if (m_Tangents)
		return;

	if (!m_Tangents)
		m_Tangents = new glm::vec3[m_NumVertices];
	else
		return;
	for (unsigned int i = 0; i < m_NumVertices; ++i)
		m_Tangents[i] = glm::vec3(0, 0, 0);

	if (m_Indices)
	{
		for (unsigned int i = 0; i < m_NumIndices; i += 3)
		{
			int a = m_Indices[i];
			int b = m_Indices[i + 1];
			int c = m_Indices[i + 2];

			glm::vec3 tangent = GenerateTangent(m_Vertices[a], m_Vertices[b], m_Vertices[c], m_TextureCoords[a], m_TextureCoords[b], m_TextureCoords[c]);

			m_Tangents[a] += tangent;
			m_Tangents[b] += tangent;
			m_Tangents[c] += tangent;
		}
	}
	else
	{
		for (unsigned int i = 0; i < m_NumVertices; i += 3)
		{
			glm::vec3 tangent = GenerateTangent(m_Vertices[i], m_Vertices[i + 1], m_Vertices[i + 2], m_TextureCoords[i], m_TextureCoords[i + 1], m_TextureCoords[i + 2]);

			m_Tangents[i] += tangent;
			m_Tangents[i + 1] += tangent;
			m_Tangents[i + 2] += tangent;
		}
	}
	for (unsigned int i = 0; i < m_NumVertices; ++i)
		glm::normalize(m_Tangents[i]);
}

glm::vec3 Mesh::GenerateTangent(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec2& ta, const glm::vec2& tb, const glm::vec2& tc)
{
	glm::vec2 coord1 = tb - ta;
	glm::vec2 coord2 = tc - ta;

	glm::vec3 vertex1 = b - a;
	glm::vec3 vertex2 = c - a;

	glm::vec3 axis = glm::vec3(vertex1*coord2.y - vertex2*coord1.y);

	float factor = 1.0f / (coord1.x * coord2.y - coord2.x * coord1.y);

	return axis * factor;
}

void Mesh::BufferData(bool multiDraw)
{
	glGenVertexArrays(1, &arrayObject);

	//GenerateNormals();
	//GenerateTangents();

	glBindVertexArray(arrayObject);

	//Buffer vertex data
	glGenBuffers(1, &bufferObject[VERTEX_BUFFER]);
	glBindBuffer(GL_ARRAY_BUFFER, bufferObject[VERTEX_BUFFER]);
	glBufferData(GL_ARRAY_BUFFER, m_NumVertices * sizeof(glm::vec3), m_Vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(VERTEX_BUFFER, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
	glEnableVertexAttribArray(VERTEX_BUFFER);

	//Buffer texture data
	/*if (m_TextureCoords)
	{
		glGenBuffers(1, &bufferObject[TEXTURE_BUFFER]);
		glBindBuffer(GL_ARRAY_BUFFER, bufferObject[TEXTURE_BUFFER]);
		glBufferData(GL_ARRAY_BUFFER, m_NumVertices * sizeof(Vec2Graphics), m_TextureCoords, GL_STATIC_DRAW);
		glVertexAttribPointer(TEXTURE_BUFFER, 2, GL_FLOAT, GL_FALSE, sizeof(Vec2Graphics), 0);
		glEnableVertexAttribArray(TEXTURE_BUFFER);
	}

	//Buffer normal data
	if (m_Normals)
	{
		glGenBuffers(1, &bufferObject[NORMAL_BUFFER]);
		glBindBuffer(GL_ARRAY_BUFFER, bufferObject[NORMAL_BUFFER]);
		glBufferData(GL_ARRAY_BUFFER, m_NumVertices * sizeof(Vec3Graphics), m_Normals, GL_STATIC_DRAW);
		glVertexAttribPointer(NORMAL_BUFFER, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3Graphics), 0);
		glEnableVertexAttribArray(NORMAL_BUFFER);
	}

	//Buffer tangent data
	if (m_Tangents)
	{
		glGenBuffers(1, &bufferObject[TANGENT_BUFFER]);
		glBindBuffer(GL_ARRAY_BUFFER, bufferObject[TANGENT_BUFFER]);
		glBufferData(GL_ARRAY_BUFFER, m_NumVertices * sizeof(Vec3Graphics), m_Tangents, GL_STATIC_DRAW);
		glVertexAttribPointer(TANGENT_BUFFER, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3Graphics), 0);
		glEnableVertexAttribArray(TANGENT_BUFFER);
	}*/

	//buffer index data
	if (m_Indices)
	{
		glGenBuffers(1, &bufferObject[INDEX_BUFFER]);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferObject[INDEX_BUFFER]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_NumIndices * sizeof(unsigned int), m_Indices, GL_STATIC_DRAW);
	}

	if (multiDraw)
	{
		for (unsigned int i = 0; i < NUM_BOIDS; ++i)
		{
			multiDrawArray[i].vertexCount = m_NumVertices;
			multiDrawArray[i].instanceCount = 1;
			multiDrawArray[i].firstIndex = 0;
			multiDrawArray[i].baseVertex = 0;
			multiDrawArray[i].baseInstance = i;
		}
		glGenBuffers(1, &bufferObject[INDIRECT_BUFFER]);
		glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bufferObject[INDIRECT_BUFFER]);
		glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(multiDrawArray), multiDrawArray, GL_STATIC_DRAW);
		//std::cout << glewGetErrorString(glGetError()) << std::endl;
	}

	Clean();

	for (auto& child : m_Children)
		child->BufferData();

	glBindVertexArray(0);
}

void Mesh::Draw()
{
	glBindVertexArray(arrayObject);
	if (bufferObject[INDIRECT_BUFFER])
	{
		glMultiDrawElementsIndirect(GL_TRIANGLE_STRIP, GL_UNSIGNED_INT, 0, NUM_BOIDS, 0);
	}
	else
	{
		if (bufferObject[INDEX_BUFFER])
		{
			glDrawElements(m_PrimitiveType, m_NumIndices, GL_UNSIGNED_INT, 0);
		}
		else
		{
			glDrawArrays(m_PrimitiveType, 0, m_NumVertices);
		}
	}

	for (auto child : m_Children)
		child->Draw();
}