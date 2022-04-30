#include "trtctx.hpp"
#include "iohelper.hpp"
#include <numeric>

class Logger : public nvi::ILogger {
public:
	void log(Severity severity, const char* msg) noexcept override {
		if ((severity == Severity::kERROR) ||
			(severity == Severity::kINTERNAL_ERROR)) {
			LOG(WARNING) << msg;
		}
	}
};

nvinfer1::ILogger& GetNVLogger(){
	static Logger s_TrtLogger;
	return s_TrtLogger;
}

int64_t ShapeVolume(const tshape &shape) {
	if (shape.empty()) {
		return 0;
	}
	int64_t nVol = 1;
	for (auto d: shape) {
		CHECK_GE(d, 0);
		nVol *= d;
	}
	return nVol;
}

tshape Dims2Shape(const nvi::Dims &dims) {
	tshape shape;
	for (int32_t i = 0; i < dims.nbDims; ++i) {
		shape.push_back(dims.d[i]);
	}
	return shape;
}

TRTContext::TRTContext(uint32_t nGpuID)
		: m_nGpuID(nGpuID)
		, m_pCudaStream(nullptr, TrtDestroyer<CUstream_st>())
		, m_pNvRuntime(nullptr, TrtDestroyer<nvi::IRuntime>())
		, m_pEngine(nullptr, TrtDestroyer<nvi::ICudaEngine>()) {
	CHECK_EQ(::cudaSetDevice(m_nGpuID), cudaSuccess);
	// Create CUDA stream
	cudaStream_t pCudaStream = nullptr;
	CHECK_EQ(cudaStreamCreate(&pCudaStream), 0);
	CHECK_NOTNULL(pCudaStream);
	m_pCudaStream.reset(pCudaStream);

	// Intialize TensorRT plugins and runtime
	CHECK(initLibNvInferPlugins(&GetNVLogger(), ""));
	m_pNvRuntime.reset(nvi::createInferRuntime(GetNVLogger()));
	CHECK_NOTNULL(m_pNvRuntime);
}

TRTContext::~TRTContext() {
	CHECK_EQ(::cudaSetDevice(m_nGpuID), cudaSuccess);
	for (auto &buf : m_InputBuffers) {
		CHECK_EQ(cudaFree(buf.second.second), cudaSuccess);
	}
	for (auto &buf : m_OutputBuffers) {
		CHECK_EQ(cudaFree(buf.second.second), cudaSuccess);
	}
}

void TRTContext::LoadModel(const std::string &strModelFile) {
	CHECK_EQ(::cudaSetDevice(m_nGpuID), cudaSuccess);

	std::string strModelBuf;
	CHECK(LoadFileContent(strModelFile, strModelBuf)) << strModelFile;

	m_pEngine.reset(m_pNvRuntime->deserializeCudaEngine(
			strModelBuf.data(), strModelBuf.size()));
	CHECK_NOTNULL(m_pEngine);

	m_pContext.reset(m_pEngine->createExecutionContext());
	CHECK_NOTNULL(m_pContext);

	std::map<std::string, tshape> inputShapes;
	for (int32_t i = 0; i < m_pEngine->getNbBindings(); ++i) {
		CHECK(m_pEngine->getBindingDataType(i) == nvi::DataType::kFLOAT);
		if (m_pEngine->bindingIsInput(i)) {
			std::string strName = m_pEngine->getBindingName(i);
			auto dims = m_pContext->getBindingDimensions(i);
			m_InputBuffers[strName] = __CreateBuf(Dims2Shape(dims));
		}
	}
	for (int32_t i = 0; i < m_pEngine->getNbBindings(); ++i) {
		std::string strName = m_pEngine->getBindingName(i);
		if (m_pEngine->bindingIsInput(i)) {
			m_Bindings.push_back(GetInputBuffer(strName).second);
		} else {
			auto dims = m_pContext->getBindingDimensions(i);
			m_OutputBuffers[strName] = __CreateBuf(Dims2Shape(dims));
			m_Bindings.push_back(GetOutputBuffer(strName).second);
		}
	}
}

std::vector<std::string> TRTContext::GetInputNames() const {
	std::vector<std::string> inputNames;
	for (auto &buf : m_InputBuffers) {
		inputNames.push_back(buf.first);
	}
	return inputNames;
}

std::vector<std::string> TRTContext::GetOutputNames() const {
	std::vector<std::string> outputNames;
	for (auto &buf : m_OutputBuffers) {
		outputNames.push_back(buf.first);
	}
	return outputNames;
}

TRTContext::GPU_MEM& TRTContext::GetInputBuffer(const std::string &strName) {
	return m_InputBuffers.find(strName)->second;
}

const TRTContext::GPU_MEM& TRTContext::GetInputBuffer(const std::string &strName) const {
	return m_InputBuffers.find(strName)->second;
}

TRTContext::GPU_MEM& TRTContext::GetOutputBuffer(const std::string &strName) {
	return m_OutputBuffers.find(strName)->second;
}

const TRTContext::GPU_MEM& TRTContext::GetOutputBuffer(const std::string &strName) const {
	return m_OutputBuffers.find(strName)->second;
}

void TRTContext::Inference() {
	CHECK_NOTNULL(m_pContext);
	std::lock_guard<std::mutex> locker(m_InferLock);
	CHECK_EQ(::cudaSetDevice(m_nGpuID), cudaSuccess);
	CHECK(m_pContext->executeV2(m_Bindings.data()));
}

TRTContext::GPU_MEM TRTContext::__CreateBuf(const tshape &shape) const {
	CHECK_EQ(::cudaSetDevice(m_nGpuID), cudaSuccess);
	GPU_MEM gpuBuf(shape, nullptr);
	auto nBufBytes = ShapeVolume(shape) * sizeof(float);
	CHECK_GT(nBufBytes, 0);
	CHECK_EQ(cudaMalloc(&gpuBuf.second, nBufBytes), 0);
	return gpuBuf;
}

void Softmax(std::vector<float> &vals) {
	for (auto &v: vals) {
		v = std::exp(v);
	}
	auto fSum = std::accumulate(vals.begin(), vals.end(), 0.f);
	fSum = std::max(fSum, 1e-4f);
	for (auto &v: vals) {
		v /= fSum;
	}
}