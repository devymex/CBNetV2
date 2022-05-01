#include "src/trtctx.hpp"
#include "src/iohelper.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <NvInfer.h>
#pragma GCC diagnostic pop
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <iostream>
#include <fstream>
#include <regex>

namespace nvi = nvinfer1;
namespace onnx = nvonnxparser;

template<typename _Stream>
_Stream& operator << (_Stream &stream, const nvinfer1::Dims &dims) {
	stream << "(";
	for (int32_t j = 0; j < dims.nbDims; ++j){
		if (j > 0) {
			stream << ", ";
		}
		stream << dims.d[j];
	}
	stream << ")";
	return stream;
}

void LoadTensorBinFile(const std::string &strFilename, TRTContext::GPU_MEM &mem) {
	std::string strBuf;
	LoadFileContent(strFilename, strBuf);
	CHECK_GT(strBuf.size(), 4);
	uint8_t *pBuf = (uint8_t*)strBuf.data();

	auto nDims = *(int32_t*)pBuf;
	CHECK_GT(strBuf.size(), (nDims + 1) * sizeof(int32_t));
	pBuf += sizeof(int32_t);

	tshape shape;
	for (int32_t d = 0; d < nDims; ++d) {
		shape.push_back(*(int32_t*)pBuf);
		pBuf += sizeof(int32_t);
	}
	auto nMemBytes = ShapeVolume(shape) * sizeof(float);
	CHECK_EQ(strBuf.size(), (nDims + 1) * sizeof(int32_t) + nMemBytes);
	
	if (mem.first.empty()) {
		CHECK(mem.second == nullptr);
		CHECK_EQ(::cudaMalloc(&mem.second, nMemBytes), cudaSuccess);
	} else {
		CHECK(mem.first == shape);
	}
	CHECK_EQ(::cudaMemcpy(mem.second, (void*)pBuf, nMemBytes,
		::cudaMemcpyHostToDevice), cudaSuccess);
}

int main(int nArgCnt, char *ppArgs[]) {
	if (nArgCnt < 2) {
		std::cout << "insufficient argument" << std::endl;
		return -1;
	}
	json jConf;
	if (ppArgs[1][0] == '{') {
		jConf = json::parse(ppArgs[1]);
	} else {
		jConf = LoadJsonFile(ppArgs[1]);
	}

	std::string strTrtFilename = std::string(jConf["trt_model"]);
	std::string strDataPath = std::string(jConf["data_path"]);
	CHECK(stdfs::is_directory(strDataPath));
	auto nGpuID = (int32_t)jConf["gpu_id"];

	CHECK_EQ(::cudaSetDevice(nGpuID), cudaSuccess);
	TRTContext trtModel { (uint32_t)nGpuID };
	trtModel.LoadModel(strTrtFilename);

	std::string strPattern = "^(.*)\\.in";
	auto inputFiles = EnumerateFiles(strDataPath, strPattern);
	auto inputNames = trtModel.GetInputNames();

	std::regex pattern(strPattern);
	std::smatch matches;
	for (auto &strInName: inputNames) {
		auto iFound = std::find_if(inputFiles.begin(), inputFiles.end(),
			[&](const std::string &strFilename){
				auto strStem = stdfs::path(strFilename).filename().string();
				std::regex_match(strStem, matches, pattern);
				strStem = matches.str(1);
				return strInName == strStem;
			});
		CHECK(iFound != inputFiles.end());
		LoadTensorBinFile(*iFound, trtModel.GetInputBuffer(strInName));
	}

	trtModel.Inference();

	if (jConf.contains("test_cnt")) {
		auto nTestCnt = (uint32_t)jConf["test_cnt"];
		uint64_t nNanoSecs = 0;
		for (uint32_t i = 0; i < nTestCnt; ++i) {
			auto nBeg = GetNowTimeNS();
			trtModel.Inference();
			nNanoSecs += GetNowTimeNS() - nBeg;
		}
		auto dAvgInfTime = (double)nNanoSecs / nTestCnt / 1000. / 1000. / 1000.;
		LOG(INFO) << "Average Inference Time: " << dAvgInfTime << "s";
	}

	std::string strOutData;
	for (auto strName: trtModel.GetOutputNames()) {
		auto outputBuf = trtModel.GetOutputBuffer(strName);
		strOutData.resize(ShapeVolume(outputBuf.first) * sizeof(float));
		CHECK_EQ(::cudaMemcpy((void*)strOutData.data(), outputBuf.second,
				strOutData.size(), ::cudaMemcpyDeviceToHost), cudaSuccess);
		auto filename = stdfs::path(strDataPath) / (strName + ".out");
		std::ofstream outFile(filename, std::ios::binary);
		int32_t nTmp = (int32_t)outputBuf.first.size();
		outFile.write((char*)&nTmp, sizeof(nTmp));
		for (auto d: outputBuf.first) {
			nTmp = (int32_t)d;
			outFile.write((char*)&nTmp, sizeof(nTmp));
		}
		outFile.write(strOutData.data(), strOutData.size());
	}
	return 0;
}