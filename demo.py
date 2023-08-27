import os
import argparse
import cv2
import numpy as np
from libs.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", type=str, default="./media/ori/270_480_video.mp4"
)
parser.add_argument(
    "--model", type=str, default="x4_270_480.trt"
)
parser.add_argument(
    "--framerate", type=int, default=30
)
parser.add_argument(
    "--height", type=int, default=270
)
parser.add_argument(
    "--width", type=int, default=480
)

def preprocess(x:np.ndarray):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, [2, 0, 1])
    x = x.astype(np.float32) / 255.0    # Normalization
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x

def postprocess(x:np.ndarray):
    x = (x * 255.0).astype(np.uint8)
    x = np.transpose(x, [1, 2, 0])
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x    

def bicubicResize(x:np.ndarray, scale:int=4):
    h, w, _ = x.shape
    x = cv2.resize(x, dsize=(w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    return x

def horizontalFusion(bi:np.ndarray, sr:np.ndarray):
    assert bi.shape == sr.shape
    h, w, c = bi.shape
    canvas = np.zeros_like(bi).astype(np.uint8)
    canvas[:, 0:w//2, :] = bi[:, 0+500:w//2+500, :]
    canvas[:, w//2:w, :] = sr[:, 0+500:w//2+500, :]
    return canvas

if __name__ == "__main__":
    opt = parser.parse_args()
    try:
        cap = cv2.VideoCapture(opt.video)
    except:
        raise ValueError(f"Failed to open video file")
    
    model_path = opt.model
    size = opt.height, opt.width
    
    # load model
    trt_model = edgeSR_TRT_Engine(
        engine_path=model_path, scale=4, lr_size=size
    )
    
    frameRate = opt.framerate

    LR_WINDOW = "LR_WINDOW"
    BICUBIC_SR_WINDOW = "BICUBIC vs SUPER-RESOLUTION"

    cv2.namedWindow(LR_WINDOW)
    cv2.namedWindow(BICUBIC_SR_WINDOW)
    cv2.moveWindow(LR_WINDOW, 80, 250)
    cv2.moveWindow(BICUBIC_SR_WINDOW, 570, 250)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bicubic = bicubicResize(frame)
        input_np = preprocess(frame)
        sr_np = postprocess(trt_model(input_np))
        key = cv2.waitKey(frameRate)
        if key == 27:
            break
        
        # Left(BICUBIC) + Right(SuperResolution) ...
        canvas = horizontalFusion(bicubic, sr_np)

        cv2.imshow(BICUBIC_SR_WINDOW, canvas)        
        cv2.imshow(LR_WINDOW, frame)
        
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
