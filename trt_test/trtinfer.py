import os, torch, tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference(object):
    def __init__(self, model_file):
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        self.trt_runtime = trt.Runtime(TRT_LOGGER)

        assert os.path.exists(model_file)
        engine_data = open(model_file, 'rb').read()
        self.trt_engine = self.trt_runtime.deserialize_cuda_engine(engine_data)
        self.context = self.trt_engine.create_execution_context()
        self.input_buffers = {}
        self.output_buffers = {}

        # for i in range(self.trt_engine.num_bindings):
        #     bname = self.trt_engine.get_binding_name(i)
        #     bshape = self.trt_engine.get_binding_shape(index=i)
        #     print(bname, bshape)

    def inference(self, inputs, stream):
        bindings = []
        device = torch.device("cuda")
        for i in range(self.trt_engine.num_bindings):
            bname = self.trt_engine.get_binding_name(i)
            bshape = self.trt_engine.get_binding_shape(index=i)
            # print(bname, bshape)
            if self.trt_engine.binding_is_input(index=i):
                assert list(bshape) == list(inputs[bname].shape)
                if inputs[bname].device == device and inputs[bname].is_contiguous():
                    self.input_buffers[bname] = inputs[bname]
                else:
                    self.input_buffers[bname] = inputs[bname].to(device,
                        memory_format=torch.contiguous_format)
                bindings.append(self.input_buffers[bname].data_ptr())
            else:
                if bname not in self.output_buffers:
                    self.output_buffers[bname] = torch.empty(size=list(bshape),
                        dtype=torch.float32, device=device).contiguous()
                bindings.append(self.output_buffers[bname].data_ptr())
        self.context.execute_async_v2(bindings, stream.cuda_stream)
        return self.output_buffers

