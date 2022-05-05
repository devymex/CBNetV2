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

	std::string strOnnxFilename = std::string(jConf["in_onnx"]);
	std::string strTrtFilename = std::string(jConf["out_trt"]);
	CHECK(stdfs::is_regular_file(strOnnxFilename)) << strOnnxFilename;

	std::map<std::string, tshape> dynInputs;
	if (jConf.contains("inputs")) {
		for (auto jTensor: jConf["inputs"]) {
			auto strName = std::string(jTensor["name"]);
			auto jShape = jTensor["shape"];
			tshape shape(jShape.size(), -1);
			for (uint32_t i = 0; i < shape.size(); ++i) {
				int32_t nInDim = int32_t(jShape[i]);
				if (nInDim > 0) {
					shape[i] = nInDim;
				}
			}
			dynInputs[std::move(strName)] = std::move(shape);
		}
	}

	// LOG(INFO) << "Preparing device for converting...";
	auto nGpuID = (int32_t)jConf["gpu_id"];
	CHECK_EQ(::cudaSetDevice(nGpuID), cudaSuccess);

	std::unique_ptr<nvi::IBuilder, TrtDestroyer<nvi::IBuilder>> pBuilder(
			nvi::createInferBuilder(GetNVLogger()));
	CHECK_NOTNULL(pBuilder);

	const auto nFlag = 1U << static_cast<uint32_t>(
			nvi::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	std::unique_ptr<nvi::INetworkDefinition, TrtDestroyer<nvi::INetworkDefinition>> pNetwork(
			pBuilder->createNetworkV2(nFlag));
	CHECK_NOTNULL(pNetwork);

	std::unique_ptr<onnx::IParser, TrtDestroyer<onnx::IParser>> pParser(
			onnx::createParser(*pNetwork, GetNVLogger()));
	CHECK_NOTNULL(pParser);

	// LOG(INFO) << "Loading ONNX model \"" << strOnnxFilename << "\"...";
	CHECK(pParser->parseFromFile(strOnnxFilename.c_str(), 5));
	for (int i = 0; i < pParser->getNbErrors(); ++i) {
		// LOG(INFO) << pParser->getError(i)->desc();
	}
	CHECK_GE(pNetwork->getNbOutputs(), 1);
	CHECK_GE(pNetwork->getNbInputs(), 1);

	// std::cout << "Layers Summary" << std::endl;
	// for (int i = 0; i < pNetwork->getNbLayers(); ++i) {
	// 	auto pLayer = pNetwork->getLayer(i);
	// 	std::ostringstream oss;
	// 	oss << "[" << i << "] \"" << pLayer->getName() << "\" has "
	// 		<< pLayer->getNbOutputs() << " outputs";
	// 	if (pLayer->getNbOutputs() > 0) {
	// 		oss << ":[";
	// 		for (int j = 0; j < pLayer->getNbOutputs(); ++j) {
	// 			oss << "" << j << ": \"" << pLayer->getOutput(j)->getName() << "\"";
	// 			if (j != pLayer->getNbOutputs() - 1) {
	// 				oss << ", ";
	// 			}
	// 		}
	// 		oss << "]";
	// 	}
	// 	std::cout << oss.str() << std::endl;
	// }
	// std::cout << std::string(70, '-') << std::endl;

	if (jConf.contains("mark_outputs")) {
		for (auto &jMarkOut: jConf["mark_outputs"]) {
			auto iLayer = int(jMarkOut["i_layer"]);
			auto iOutput = int(jMarkOut["i_output"]);
			auto pMarkOut = pNetwork->getLayer(iLayer)->getOutput(iOutput);
			pNetwork->markOutput(*pMarkOut);
		}
	}

	if (jConf.contains("unmark_outputs")) {
		for (auto &jMarkOut: jConf["unmark_outputs"]) {
			auto iLayer = int(jMarkOut["i_layer"]);
			auto iOutput = int(jMarkOut["i_output"]);
			auto pMarkOut = pNetwork->getLayer(iLayer)->getOutput(iOutput);
			pNetwork->unmarkOutput(*pMarkOut);
		}
	}

	// LOG(INFO) << "Converting...";
	std::unique_ptr<nvi::IBuilderConfig, TrtDestroyer<nvi::IBuilderConfig>> pConfig(
			pBuilder->createBuilderConfig());
	CHECK_NOTNULL(pConfig);
	pConfig->setMaxWorkspaceSize(1 << 30);
	if (jConf.contains("data_type")) {
		auto strDataType = std::string(jConf["data_type"]);
		if (strDataType == "fp16") {
			pConfig->setFlag(nvi::BuilderFlag::kFP16);
		} else if (strDataType == "int8") {
			LOG(FATAL) << "Unsupported data type: " << strDataType;
		} else if (strDataType == "fp32") {
		} else {
			LOG(FATAL) << "Unsupported data type: " << strDataType;
		}
		// LOG(INFO) << "Data type of computation: " << strDataType;
	}

	for (int32_t i = 0; i < pNetwork->getNbInputs(); ++i) {
		auto pName = pNetwork->getInput(i)->getName();
		auto oriDims = pNetwork->getInput(i)->getDimensions();
		CHECK_GT(oriDims.nbDims, 0);
		auto dims = oriDims;

		for (int32_t j = 0; j < dims.nbDims; ++j) {
			CHECK_NE(dims.d[j], 0);
			if (dims.d[j] < 0) {
				auto iInput = dynInputs.find(pName);
				CHECK(iInput != dynInputs.end()) << pName;
				CHECK_GT(iInput->second.size(), j);
				CHECK_GT(iInput->second[j], 0);
				dims.d[j] = iInput->second[j];
			}
		}
		pNetwork->getInput(i)->setDimensions(dims);
		// std::ostringstream oss;
		// oss << "Input " << i << ": \"" << pName << "\" " << oriDims << " -> " << dims;
		// LOG(INFO) << oss.str();
	}

#if NV_TENSORRT_MAJOR < 8
	std::unique_ptr<nvi::ICudaEngine, TrtDestroyer<nvi::ICudaEngine>> pEngine(
			pBuilder->buildEngineWithConfig(*pNetwork, *pConfig));
#else
	std::unique_ptr<nvi::IHostMemory, TrtDestroyer<nvi::IHostMemory>> pNetMem(
			pBuilder->buildSerializedNetwork(*pNetwork, *pConfig));
	CHECK_NOTNULL(pNetMem);
	std::unique_ptr<nvi::IRuntime, TrtDestroyer<nvi::IRuntime>> pRuntime{
			nvi::createInferRuntime(GetNVLogger())};
	CHECK_NOTNULL(pRuntime);
	std::unique_ptr<nvi::ICudaEngine, TrtDestroyer<nvi::ICudaEngine>> pEngine(
		pRuntime->deserializeCudaEngine(pNetMem->data(), pNetMem->size()));
#endif
	CHECK_NOTNULL(pEngine);

	// for (int32_t i = 0; i < pEngine->getNbBindings(); ++i) {
	// 	std::ostringstream oss; 
	// 	oss << "Binding " << i << "["
	// 		<< (pEngine->bindingIsInput(i) ? "I" : "O") << "]: \""
	// 		<< pEngine->getBindingName(i) << "\" "
	// 		<< pEngine->getBindingDimensions(i);
	// 	LOG(INFO) << oss.str();
	// }

	std::unique_ptr<nvi::IHostMemory, TrtDestroyer<nvi::IHostMemory>> pHostMem(
			pEngine->serialize());
	CHECK_NOTNULL(pHostMem);

	std::ofstream out_stream(strTrtFilename, std::ios::binary);
	CHECK(out_stream.is_open());
	out_stream.write((const char*)pHostMem->data(), pHostMem->size());
	out_stream.close();

	// LOG(INFO) << "Done with \"" << strTrtFilename << "\".";
	return 0;
}