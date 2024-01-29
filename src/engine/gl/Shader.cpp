#include <engine/Shader.h>
#include <stdexcept>
#include <iostream>

Shader::Shader(const std::string& name) 
	:
	_name(name),
	_program(0)
{}

Shader::~Shader()
{
	glDeleteProgram(_program);
}

void Shader::Load(const std::string & vertexSource, const std::string& fragmentSource)
{
	GLuint vertexShader = loadShader(vertexSource, GL_VERTEX_SHADER);
	GLuint fragmentShader = loadShader(fragmentSource, GL_FRAGMENT_SHADER);

	createShaderProgram(vertexShader, fragmentShader);
	findAttributes();
	findUniforms();

	// Delete the shaders as they're linked into our program now and no longer necessary
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

void Shader::Use()
{
	glUseProgram(_program);
}

GLint Shader::GetAttributeLocation(const std::string& name) const
{
	auto it = _attributes.find(name);
	if (it == _attributes.end()) 
	{
		throw std::runtime_error("Attribute not found: " + name);
	}

	return it->second;
}

GLint Shader::GetUniformLocation(const std::string& name) const
{
	auto it = _uniforms.find(name);
	if (it == _uniforms.end()) {
		throw std::runtime_error("Uniform not found: " + name);
	}
	return it->second;
}

GLuint Shader::loadShader(const std::string& source, GLenum type)
{
	GLuint shader = glCreateShader(type);
	const char* src = source.c_str();
	glShaderSource(shader, 1, &src, 0);
	glCompileShader(shader);

	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetShaderInfoLog(shader, 512, 0, infoLog);
		std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
		throw std::runtime_error("Failed to compile shader " + _name);
	}

	return shader;
}

void Shader::createShaderProgram(GLuint vertexShader, GLuint fragmentShader)
{
	_program = glCreateProgram();
	glAttachShader(_program, vertexShader);
	glAttachShader(_program, fragmentShader);
	glLinkProgram(_program);

	GLint success;
	glGetProgramiv(_program, GL_LINK_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetProgramInfoLog(_program, 512, 0, infoLog);
		std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		throw std::runtime_error("Faield to link shader " + _name);
	}
}

void Shader::findAttributes()
{
	int count;
	glGetProgramiv(_program, GL_ACTIVE_ATTRIBUTES, &count);

	for (int i = 0; i < count; i++)
	{
		char attributeName[1024];
		GLsizei attributeNameLength;
		GLint attributeSize;
		GLenum attributeType;
		glGetActiveAttrib(_program, i, sizeof(attributeName), &attributeNameLength, &attributeSize, &attributeType, attributeName);

		GLint attributeLocation = glGetAttribLocation(_program, attributeName);
		_attributes[attributeName] = attributeLocation;
	}
}

void Shader::findUniforms()
{
	int count;
	glGetProgramiv(_program, GL_ACTIVE_UNIFORMS, &count);

	for (GLint i = 0; i < count; ++i) {
		char uniformName[1024];
		GLsizei uniformNameLength;
		GLint uniformSize;
		GLenum uniformType;
		glGetActiveUniform(_program, i, sizeof(uniformName), &uniformNameLength, &uniformSize, &uniformType, uniformName);

		GLint uniformLocation = glGetUniformLocation(_program, uniformName);
		_uniforms[uniformName] = uniformLocation;
	}
}

