#ifndef _TRTCTX_HPP
#define _TRTCTX_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <NvInferPlugin.h>
#include <NvInfer.h>
#pragma GCC diagnostic pop

#include <cuda_runtime.h>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nvi = nvinfer1;

template<typename T>
struct TrtDestroyer {
	void operator()(T *t) {
#if NV_TENSORRT_MAJOR <= 7
		t->destroy();
#endif // NV_TENSORRT_MAJOR <= 7
	}
};

template<>
struct TrtDestroyer<CUstream_st> {
	void operator()(cudaStream_t s) {
		cudaStreamDestroy(s);
	}
};

using TENSOR_SHAPE = std::vector<int32_t>;

int64_t ShapeVolume(const TENSOR_SHAPE &shape);

class TRTContext {
public:
	using GPU_MEM = std::pair<TENSOR_SHAPE, void*>;

private:
	const uint32_t m_nGpuID = 0;

	std::map<std::string, GPU_MEM> m_InputBuffers;
	std::map<std::string, GPU_MEM> m_OutputBuffers;

	std::unique_ptr<CUstream_st, TrtDestroyer<CUstream_st>> m_pCudaStream;
	std::unique_ptr<nvi::IRuntime, TrtDestroyer<nvi::IRuntime>> m_pNvRuntime;
	std::unique_ptr<nvi::ICudaEngine, TrtDestroyer<nvi::ICudaEngine>> m_pEngine;
	std::unique_ptr<nvi::IExecutionContext, TrtDestroyer<nvi::IExecutionContext>> m_pContext;

	std::mutex m_InferLock;
	std::vector<void*> m_Bindings;

public:
	TRTContext(uint32_t nGpuID);

	virtual ~TRTContext();

	void LoadModel(const std::string &strModelFile,
			const std::map<std::string, TENSOR_SHAPE> &inputShapes);

	std::vector<std::string> GetInputNames() const;

	std::vector<std::string> GetOutputNames() const;

	GPU_MEM& GetInputBuffer(const std::string &strName);

	const GPU_MEM& GetInputBuffer(const std::string &strName) const;

	GPU_MEM& GetOutputBuffer(const std::string &strName);

	const GPU_MEM& GetOutputBuffer(const std::string &strName) const;

	void Inference();

private:
	GPU_MEM __CreateBuf(const TENSOR_SHAPE &shape) const;
};

nvinfer1::ILogger& GetNVLogger();

void Softmax(std::vector<float> &vals);

#endif // #ifndef _TRT_CTX_HPP