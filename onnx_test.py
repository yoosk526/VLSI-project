import onnxruntime as ort
import numpy as np
import cv2
from libs.utils import *

def bicubicResize(x:np.ndarray, scale:int=4):
	h, w, _ = x.shape
	x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
	x = cv2.resize(x, dsize=(w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
	return x

lrOrig = openImage("./media/ar_02.png")
lrObj = np.transpose(lrOrig, [2, 0, 1])
lrObj = np.expand_dims(lrObj, axis=0).astype(np.float32) / 255.0
ort_sess = ort.InferenceSession("./model/x4_224_320.onnx", providers=['CUDAExecutionProvider'])
outputs = (ort_sess.run(None, {'input': lrObj})[0] * 255.0).astype(np.uint8)
srObj = np.squeeze(outputs, axis=0)
srObj = np.transpose(srObj, [1, 2, 0])
srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)

# cv2.imwrite("result.png", srObj)
    
bicubic = bicubicResize(lrOrig)
total_img = np.hstack((bicubic, srObj))

# cv2.imwrite("./media/x4_bicubic_02.png", bicubic)
cv2.imwrite("./media/x4_abpn_02.png", srObj)

'''
while True:
    cv2.imshow("result", total_img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
'''
