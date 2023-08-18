import cv2
import numpy as np
from libs.utils import *

if __name__ == "__main__":
    trt_model = edgeSR_TRT_Engine(
        engine_path="./model/x4_270_480.trt", scale=4, lr_size=(270, 480)
    )
    lrOrig = openImage("./media/aerial_ku.png")
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
