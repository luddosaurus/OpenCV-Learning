import cv2
print("Package Imported")


img = cv2.imread("res/lena.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Output", imgGray)
cv2.waitKey(0)