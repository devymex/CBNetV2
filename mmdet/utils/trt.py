import torch
from typing import List

def export_onnx(model: torch.nn.Module, input_shapes: dict, output_names: List[str], onnx_filename: str):
    model.eval()
    input_names = list(input_shapes.keys())

    sample_inputs = []
    dynamic_axes = {}
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
