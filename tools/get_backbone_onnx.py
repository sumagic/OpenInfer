import torch
import torchvision.models as models
import torch.onnx
import onnx
import onnxoptimizer


model = models.resnet101(pretrained=True)
model.eval()

input = torch.randn(1,3,640,640)
output_path = 'resnet101_3x640x640.onnx'

torch.onnx.export(model, input, output_path,export_params=True,
    opset_version=12, do_constant_folding=True, input_names=['input'],
    output_names=['output'], dynamic_axes={'input': {0: 'batch_size'},
                                           'output': {0: 'batch_size'}})

onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("Model has been converted successfully!")