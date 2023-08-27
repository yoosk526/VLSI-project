import os
import argparse
import cv2
import numpy as np
from libs.utils import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--img", type=str, default="./media/ori/270_480_img_01.png"
)
parser.add_argument(
    "--model", type=str, default="./model/visdrone_abpn/x4_270_480.trt"
)

if __name__ == "__main__":
	# opt = parser.parse_args()
	
    trt_model = edgeSR_TRT_Engine(engine_path="./model/visdrone_abpn/x4_270_480_20t.trt", scale=4, lr_size=(270, 480))
    lrOrig = openImage("./media/ori/270_480_img_01.png")
    lrObj = np.transpose(lrOrig, [2, 0, 1])
    lrObj = np.ascontiguousarray(lrObj, dtype=np.float32)
    while True:
		cv2.imshow("Low resolution", lrOrig)
		key = cv2.waitKey(1)
        if key == 27:
        	break
    cv2.destroyAllWindows()
    lrObj /= 255.0  # Normalization    
    srObj = (trt_model(lrObj) * 255.0).astype(np.uint8)
    srObj = np.transpose(srObj, [1,2,0])
    srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)
    
    while True:
        cv2.imshow("High resolution", srObj)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
