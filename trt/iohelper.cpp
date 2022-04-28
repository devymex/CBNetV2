#include "iohelper.hpp"
#include <glog/logging.h>
#include <chrono>
#include <fstream>

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
