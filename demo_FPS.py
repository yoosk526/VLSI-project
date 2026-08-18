import os
import argparse
import cv2
import numpy as np
import time
from libs.utils import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "--video", type=str, default="./media/ori/270_480.mp4"
)

parser.add_argument(
    "--model", type=str, default="./model/x4_270_480.trt"
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

parser.add_argument(
    "--scale", type=int, default=4
)

parser.add_argument(
    "--norm", action="store_true"
)

if __name__ == "__main__":
    opt = parser.parse_args()
    try:
        cap = cv2.VideoCapture(opt.video)
    except:
        raise ValueError("Failed to open video file")

    model_path = opt.model
    size = opt.height, opt.width
    upscale = opt.scale

    # Load TensorRT Model
    trt_model = edgeSR_TRT_Engine(
        engine_path=model_path, scale=upscale, lr_size=size
    )
    
    # Video FPS 설정
    frameRate = opt.framerate

    # cv2.waitKey()는 ms 단위이므로
    # FPS → delay(ms)로 변환
    delay = int(1000 / frameRate)

    # Window 설정
    LR_WINDOW = "LR_WINDOW"
    BICUBIC_SR_WINDOW = "BICUBIC vs SUPER-RESOLUTION"

    cv2.namedWindow(LR_WINDOW)
    cv2.namedWindow(BICUBIC_SR_WINDOW)
    cv2.moveWindow(LR_WINDOW, 80, 30)
    cv2.moveWindow(BICUBIC_SR_WINDOW, 570, 30)

    # FPS smoothing 변수
    smooth_fps = 0.0
    alpha = 0.1


    # Video Loop
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Bicubic Upscaling
        bicubic = bicubicResize(frame, upscale)

        # Preprocess
        input_np = preprocess(frame, opt.norm)

        # TensorRT Inference
        infer_start = time.perf_counter()
        sr_output = trt_model(input_np)
        infer_end = time.perf_counter()

        # Inference Time 계산
        infer_time = infer_end - infer_start
        infer_ms = infer_time * 1000

        if infer_time > 0:
            current_fps = 1.0 / infer_time
        else:
            current_fps = 0.0

        # FPS smoothing
        if smooth_fps == 0:
            smooth_fps = current_fps
        else:
            smooth_fps = (1 - alpha) * smooth_fps + alpha * current_fps

        # Postprocess
        sr_np = postprocess(sr_output, opt.norm)

        # Bicubic + SR
        canvas = horizontalFusion(bicubic, sr_np)

        # FPS Text 출력
        cv2.putText(canvas, f"SR FPS : {smooth_fps:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        # Inference Time 출력
        cv2.putText(canvas, f"Inference : {infer_ms:.2f} ms", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Window 출력
        cv2.imshow(BICUBIC_SR_WINDOW,canvas)
        cv2.imshow(LR_WINDOW, frame)

        # ESC 종료
        key = cv2.waitKey(delay)
        if key == 27:
            break


    # 종료 처리
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()