import os
import argparse
import onnxruntime as ort
import numpy as np
import cv2
from libs.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
	"--image", type=str, default="./media/ori/270_480_01.png"
)
parser.add_argument(
	"--model", type=str, default="~/working/train/onnx/x4_270_480.onnx"
)
parser.add_argument(
	"--save", type=str, default="./media/result/onnx_x4_270_480_01.png"
)

def bicubicResize(x:np.ndarray, scale:int=4):
	h, w, _ = x.shape
	x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
	x = cv2.resize(x, dsize=(w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
	return x

if __name__ == "__main__":
	opt = parser.parse_args()
	
	lrOrig = openImage(opt.image)
	lrObj = np.transpose(lrOrig, [2, 0, 1])
	lrObj = np.expand_dims(lrObj, axis=0).astype(np.float32) / 255.0
	
	ort_sess = ort.InferenceSession(opt.model, providers=['CUDAExecutionProvider'])
	
	outputs = (ort_sess.run(None, {'input': lrObj})[0] * 255.0).astype(np.uint8)	
	srObj = np.squeeze(outputs, axis=0)
	srObj = np.transpose(srObj, [1, 2, 0])
	srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
	
	cv2.imwrite(opt.save, srObj)