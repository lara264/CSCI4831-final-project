import numpy as np
import argparse
import time
import cv2
import os
import matplotlib as plt

# https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
def click(event, x, y, flags, params):
    if(event == cv2.EVENT_LBUTTONDOWN):
        print(x, "", y)


# https://www.geeksforgeeks.org/python-foreground-extraction-in-an-image-using-grabcut-algorithm/
image = cv2.imread("../People_Images/Person_52.png")
mask = np.zeros(image.shape[:2], dtype="uint8")

backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)

# cv2.imshow("image", image)
# cv2.setMouseCallback("image", click)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# rect = (45, 15, 313, 251)
# cv2.grabCut(image, mask, rect, backgroundModel, foregroundModel, 10, cv2.GC_INIT_WITH_RECT)
#
# mask2 = np.where((mask==2)|(mask==0), 0, 1).astype("uint8")
# image = image * mask2[:,:, np.newaxis]
#
# #idea 1
# outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
# outputMask = (outputMask * 255).astype("uint8")
# output = cv2.bitwise_and(image, image, mask=outputMask)
# cv2.imshow("output", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# idea 2
# outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
# outputMask = (outputMask * 255).astype("uint8")
# output = cv2.bitwise_and(image, image, mask=outputMask)
# cv2.imshow("output", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# idea 3
# print(image.shape[:2])
#
# minX = 136
# maxX = 232
# minY = 46
# maxY = 253
#
# mask = np.zeros(image.shape[:2], dtype="uint8")
# for i in range(minY, maxY):
#     for j in range(minX, maxX):
#         mask[i][j] = 1
#
# mask[mask > 0] = cv2.GC_FGD
#
# cv2.grabCut(image, mask, None, backgroundModel, foregroundModel, 10, cv2.GC_INIT_WITH_MASK)
# outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
# outputMask = (outputMask * 255).astype("uint8")
# output = cv2.bitwise_and(image, image, mask=outputMask)
# cv2.imshow("output", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# idea 4
minX = 136
maxX = 232
minY = 46
maxY = 253

mask = np.zeros(image.shape[:2], dtype="uint8")
known_background = np.zeros(image.shape[:2], dtype="uint8")
for i in range(minY, maxY):
    for j in range(minX, maxX):
        mask[i][j] = 1
        known_background[i][j] = 1

mask[mask > 0] = cv2.GC_FGD

rect = (45, 15, 313, 251)
mask = np.zeros(image.shape[:2], dtype="uint8")

cv2.grabCut(image, mask, rect, backgroundModel, foregroundModel, 10, cv2.GC_INIT_WITH_RECT)
outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
outputMask = (outputMask * 255).astype("uint8")
output = cv2.bitwise_and(image, image, mask=outputMask)
output = cv2.bitwise_xor(image, output, mask=known_background)
cv2.imshow("output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# idea 5
# minX = 136
# maxX = 232
# minY = 46
# maxY = 253
#
# newmask = np.zeros(image.shape[:2], dtype="uint8")
# for i in range(minY, maxY):
#     for j in range(minX, maxX):
#         newmask[i][j] = 1
#
# #mask[newmask == 0] = 0
# mask[newmask == 1] = 1
# mask, bgdModel, fgdModel = cv2.grabCut(image,mask,None,backgroundModel,foregroundModel,5,cv2.GC_INIT_WITH_MASK)
# mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = image*mask[:,:,np.newaxis]
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
