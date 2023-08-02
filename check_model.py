import sys
import onnx

filename = "./model/x4_270_480.onnx"
model = onnx.load(filename)
onnx.checker.check_model(model, full_check=True)
