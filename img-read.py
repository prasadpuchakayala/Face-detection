import cv2
img=cv2.imread("farmer.png")
print(img.shape)
img=cv2.resize(img,(250,250))
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(img,200, 225, cv2.THRESH_BINARY)
cv2.imshow("threshold",thresh1)
img = cv2.Canny(img, 100, 200)
print(img.shape)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
