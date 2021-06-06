import cv2

img = cv2.imread("/home/ankit/Documents/ProductRecognition/CODE/yolov5/coco128/images/train2017/000000000009.jpg")
#cv2.imshow("simple image",img)

CnMap = cv2.Canny(img,100,180)

(B,G,R) = cv2.split(img)
img = cv2.merge([CnMap,G,R,B])

# cv2.imshow("Canny Map",CnMap)
# cv2.imshow("Canny map superimposed img",img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(img.shape)
print(img[...,:3].shape)