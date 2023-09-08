import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
	"--img_dir", type=str, default="./media/ori/270_480_01.png"
)
parser.add_argument(
	"--save_dir", type=str, default="./media/ref/x4_270_480_bicubic_01.png"
)

def bicubicResize(x:np.ndarray, scale:int=4):
	h, w, _ = x.shape
	x = cv2.resize(x, dsize=(w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
	return x

if __name__ == "__main__":
	args = parser.parse_args()
	
	lrOrig = cv2.imread(args.img_dir)
	bicubic = bicubicResize(lrOrig)
	
	cv2.imwrite(args.save_dir, bicubic)
