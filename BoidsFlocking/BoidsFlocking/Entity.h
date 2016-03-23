#pragma once
#include"Common.h"
#include "RenderComponent.h"

class Entity
{
	friend class Scene;
public:
	Entity(const std::string& name = std::to_string(id));
	virtual ~Entity();

	Entity*	FindEntity(const std::string& name);
	void AddChildObject(Entity* child);

	inline const std::string& GetName() { return m_Name; }
	inline std::vector<Entity*>& GetChildren() { return m_Children; }

	inline void SetWorldTransform(const glm::mat4& transform) { m_WorldTransform = transform; m_WorldTransformPtr = &m_WorldTransform; }
	inline void SetWorldTransform(glm::mat4* transform) { m_WorldTransformPtr = transform; }
	inline const glm::mat4* GetWorldTransform() const { return m_WorldTransformPtr; }

	inline void SetLocalTransform(const glm::mat4& transform) { m_LocalTransform = transform; }
	inline const glm::mat4& GetLocalTransform() const { return m_LocalTransform; }

	inline void SetRenderComponent(RenderComponent* comp) { m_RenderComponent = comp; m_RenderComponent->SetParent(this); }
	inline RenderComponent* GetRenderComponent() const { return m_RenderComponent; }

protected:
	virtual void OnRenderObject();			//Handles OpenGL calls to Render the object
	virtual void OnUpdateObject(float dt);	//Override to handle things like AI etc on update loop

	std::string			 m_Name;
	Entity*				 m_Parent;
	std::vector<Entity*> m_Children;

	RenderComponent* m_RenderComponent;

	glm::mat4 m_WorldTransform;
	glm::mat4* m_WorldTransformPtr;
	glm::mat4 m_LocalTransform;

	float m_CamDist; //For ordering of rendering lists.
	static unsigned int id;
};