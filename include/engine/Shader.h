#pragma once
#include <string>
#include <glad/glad.h>
#include <unordered_map>

class Shader 
{
public:
	Shader(const std::string& name);
	~Shader();
	void Load(const std::string& vertexSource, const std::string& fragmentSource);
	void Use();
	GLint GetAttributeLocation(const std::string& name) const;
	GLint GetUniformLocation(const std::string& name) const;

private:
	std::string _name;
	GLuint _program;
	std::unordered_map<std::string, GLint> _attributes;
	std::unordered_map<std::string, GLint> _uniforms;

	GLuint loadShader(const std::string& source, GLenum type);
	void createShaderProgram(GLuint vertexShader, GLuint fragmentShader);
	void findAttributes();
	void findUniforms();
};