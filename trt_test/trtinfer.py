import os, numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class HostDeviceMem(object):
    def __init__(self, shape, host_mem, device_mem):
        self.shape = shape
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TRTInference(object):
    def __init__(self, model_file):
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        self.stream = cuda.Stream()
        self.trt_runtime = trt.Runtime(TRT_LOGGER)

        assert os.path.exists(model_file)
        engine_data = open(model_file, 'rb').read()
        self.trt_engine = self.trt_runtime.deserialize_cuda_engine(engine_data)

        self.input_buffers = {}
        self.output_buffers = {}
        self.bindings = []
        for i in range(self.trt_engine.num_bindings):
            bname = self.trt_engine.get_binding_name(i)
            bshape = self.trt_engine.get_binding_shape(index=i)
            print(bname, bshape)
            buf = self._create_buf(bshape)
            self.bindings.append(buf.device)
            if self.trt_engine.binding_is_input(index=i):
                self.input_buffers[bname] = buf
            else:
                self.output_buffers[bname] = buf

        self.context = self.trt_engine.create_execution_context()

    def inference(self, inputs):
        for name in inputs:
            buf = self.input_buffers[name]
            assert buf.shape == inputs[name].shape
            np.copyto(buf.host, inputs[name].ravel())
            cuda.memcpy_htod_async(buf.device, buf.host, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        outputs = {}
        for name in self.output_buffers:
            buf = self.output_buffers[name]
            cuda.memcpy_dtoh_async(buf.host, buf.device, self.stream)
            outputs[name] = np.reshape(buf.host, buf.shape)
        self.stream.synchronize()
        return outputs

    def _create_buf(self, shape):
        size = trt.volume(shape)
        dtype = np.float32
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        return HostDeviceMem(shape, host_mem, device_mem)