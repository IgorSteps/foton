#include <engine/File.h>
#include <fstream>
#include <sstream>
#include <iostream>

std::string FileIO::ReadFile(const std::string& path)
{
	std::string stringifiedFile;
	std::ifstream file;

	// ensure ifstream objects can throw exceptions:
	file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
		file.open(path);

		std::stringstream fileStream;
		fileStream << file.rdbuf();
		file.close();
		
		stringifiedFile = fileStream.str();
	}
	catch (std::ifstream::failure e)
	{
		std::cerr << "ERROR::FILEIO::FILE_NOT_SUCCESFULLY_READ" << std::endl;
		std::cerr << "Faied to read from file: " + path + " because: " + e.what() << std::endl;
	}

	return stringifiedFile;
}