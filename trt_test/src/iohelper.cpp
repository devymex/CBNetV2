#include "iohelper.hpp"
#include <glog/logging.h>
#include <chrono>
#include <fstream>
#include <regex>

bool LoadFileContent(const std::string &strFn, std::string &strFileBuf) {
	std::ifstream inFile(strFn, std::ios::binary);
	if (!inFile.is_open()) {
		return false;
	}
	inFile.seekg(0, std::ios::end);
	strFileBuf.resize((uint64_t)inFile.tellg());
	inFile.seekg(0, std::ios::beg);
	inFile.read(const_cast<char*>(strFileBuf.data()), strFileBuf.size());
	CHECK(inFile.good());
	return true;
}

json LoadJsonFile(const std::string &strFilename) {
	std::string strConfContent;
	CHECK(LoadFileContent(strFilename, strConfContent)) << strFilename;
	return json::parse(strConfContent);
}

int64_t GetNowTimeNS() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
}

std::vector<std::string> EnumerateFiles(const std::string &strPath,
		const std::string &strPattern) {
	std::vector<std::string> filenames;
	for (auto &entry : stdfs::recursive_directory_iterator(strPath,
			stdfs::directory_options::skip_permission_denied)) {
		auto strFilename = entry.path().filename().string();
		std::smatch match;
		if (std::regex_match(strFilename, match, std::regex(strPattern))) {
			filenames.emplace_back(entry.path().string());
		}
	}
	return filenames;
}
