import os
import argparse
import cv2
import numpy as np
from libs.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
	"--image", type=str, default="./media/ori/270_480_01.png"
)
parser.add_argument(
	"--model", type=str, default="./model/x4_270_480.trt"
)
parser.add_argument(
	"--save", type=str, default="./media/result/trt_270_480_01.png"
)
parser.add_argument(
	"--upscale", type=int, default=4
)


if __name__ == "__main__":
	opt = parser.parse_args()
		
	trt_model = edgeSR_TRT_Engine(
		engine_path=opt.model, scale=opt.upscale, lr_size=(270, 480)
	)
	lrOrig = openImage(opt.image)
	
	# View image
	'''
	while True:
		cv2.imshow("Low resoultion", lrOrig)
		key = cv2.waitKey(1)
		if key == 27:
			cv2.destroyAllWindows()
			break
	'''
	
	# SuperResolution
	lrObj = np.transpose(lrOrig, [2, 0, 1])		# H, W, C -> C, H, W
	lrObj = np.ascontiguousarray(lrObj, dtype=np.float32)	# return contiguous array
	lrObj /= 255.0
	srObj = (trt_model(lrObj) * 255.0).astype(np.uint8)
	srObj = np.transpose(srObj, [1, 2, 0])
	srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
	
	cv2.imwrite(opt.save, srObj)
