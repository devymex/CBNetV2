import torch, os, subprocess, json, array, typing, numpy as np
from trt_test.trtinfer import TRTInference

class TrtHelper:
    def __init__(self, name):
        self.name = name
        self.model = None

    def trtinfer(self, inputs, out_names, dynamic_batch = False, test_cnt = 0):
        # export onnx
        onnx_file = f'trt_test/models/{self.name}.onnx'
        if not os.path.exists(onnx_file):
            input_shapes = {}
            for name in inputs:
                input_shapes[name] = inputs[name].shape
            dynamic_axes = {}
            if dynamic_batch:
                for name in inputs:
                    dynamic_axes[name] = {0: 'batch'}
                for name in out_names:
                    dynamic_axes[name] = {0: 'batch'}
            export_onnx(self, input_shapes, out_names, dynamic_axes, onnx_file)

        # convert to trt
        trt_file = f'trt_test/models/{self.name}.trt'
        if not os.path.exists(trt_file):
            trtconv = {
                'gpu_id': 2,
                'in_onnx': onnx_file,
                'out_trt': trt_file,
                'inputs': [],
                'data_type': 'fp16'
                }
            for name in inputs:
                trtconv['inputs'].append({'name': name, 'shape': []})
            trtconv = json.dumps(trtconv)
            subprocess.run(['build/trtconv', trtconv])

        if self.model is None:
            self.model = TRTInference(trt_file)
        outputs = self.model.inference(inputs, torch.cuda.current_stream())
        # outputs = {}
        # for name in numpy_res:
        #     outputs[name] = torch.tensor(numpy_res[name])
        return outputs


        # store input data
        data_path = f'trt_test/data/{self.name}'
        os.makedirs(data_path, exist_ok=True)
        for name in inputs:
            bin_file = open(os.path.join(data_path, f'{name}.in'), 'wb')
            data = inputs[name].cpu().numpy()
            dim_shape = [data.ndim] + list(data.shape)
            array.array('i', dim_shape).tofile(bin_file)
            array.array('f', data.flatten().tolist()).tofile(bin_file)
            bin_file.close()

        # trt inference
        trtinfer = {
            'gpu_id': 1,
            'trt_model': trt_file,
            'data_path': data_path,
            'report_file': f'trt_test/results/{self.name}.prof'
            }
        if test_cnt > 0:
            trtinfer['test_cnt'] = test_cnt
        trtinfer = json.dumps(trtinfer)
        subprocess.run(['build/trtinfer', trtinfer])

        outputs = {}
        for filename in os.listdir(data_path):
            if filename.endswith('.out'):
                int.from_bytes
                buf = open(os.path.join(data_path, filename), 'rb').read()

                ndim = array.array('i')
                ndim.fromstring(buf[: 4])
                ndim = ndim[0]

                dims = array.array('i')
                dims.fromstring(buf[4 : (ndim + 1) * 4])
                dims = list(dims)

                vals = array.array('f')
                vals.fromstring(buf[(ndim + 1) * 4:])
                vals = list(vals)
                assert len(vals) == np.prod(dims)

                vals = torch.Tensor(vals).reshape(dims)
                outputs[os.path.splitext(filename)[0]] = vals
        return outputs

def export_onnx(model: torch.nn.Module, input_shapes: dict, output_names: typing.List[str],
        dynamic_axes: dict, onnx_filename: str):
    model.eval()
    input_names = list(input_shapes.keys())

    sample_inputs = []
    for name in input_names:
        sample_inputs.append(torch.zeros(input_shapes[name], requires_grad=False))
    sample_inputs = tuple(sample_inputs)

    torch.onnx.export(
        model=model.cpu(),
        args=sample_inputs,
        f=onnx_filename,
        export_params=True,
        verbose=False,
        opset_version=9,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes)
