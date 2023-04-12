from image_Stitcher import image_Stitcher
import cv2

# 读取图片
path1 = r"C:\demo\image_cat\image_code\1.jpg"
path2 = r"C:\demo\image_cat\image_code\2.jpg"
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
# 图片拼接
stitcher = image_Stitcher() # 定义函数Stitcher(),这个函数用于实现图像拼接
result, vis = stitcher.stitch([img1, img2], showMatches=True) #将图片1，图片2传入函数Stitcher()中

cv2.namedWindow("Contours", cv2.WINDOW_NORMAL) #显示第一张图片
cv2.imshow("Contours", img1)
#cv2.imshow('img1', img1)
cv2.namedWindow("img2", cv2.WINDOW_NORMAL)#显示第二张图片
cv2.imshow('img2', img2)

# cv2.namedWindow("keypoints matches", cv2.WINDOW_NORMAL)
# cv2.imshow('keypoints matches', vis)
cv2.namedWindow("result", cv2.WINDOW_NORMAL) #显示融合后的图片
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyWindow()

