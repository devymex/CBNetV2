#ifndef _IOHELPER_HPP
#define _IOHELPER_HPP

#include "json.hpp"
#include <glog/logging.h>
#include <string>

#ifdef __GNUC__
#include <experimental/filesystem>
namespace stdfs = std::experimental::filesystem;
#else // __GNUC__
#include <filesystem>
namespace stdfs = std::filesystem;
#endif // __GNUC__

using json = nlohmann::json;

bool LoadFileContent(const std::string &strFn, std::string &strFileBuf);

json LoadJsonFile(const std::string &strFilename);

int64_t GetNowTimeNS();

std::vector<std::string> EnumerateFiles(const std::string &strPath,
		const std::string &strPattern);

#endif // _IOHELPER_HPP