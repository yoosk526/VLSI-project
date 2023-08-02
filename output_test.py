import numpy as np
import cv2

img = cv2.imread('./ms3_01.png')
print(img.shape)
cv2.imshow('img', img)

cv2.waitKey()
cv2.destroyAllWindows()
