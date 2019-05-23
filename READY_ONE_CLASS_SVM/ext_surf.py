import cv2
import numpy as np 

#Extract SURF Features
def get_features(img):
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)
    print("No. of KP: " + str(len(kp)))
    print("No. of descriptors: " + str(len(des)))
    print("Size of each descriptor: " + str(len(des[0])))
    print("shape of des " + str(des.shape))
    print("flattened des: " + str(des.flatten().shape))
    return [kp, des]


img = cv2.imread("crowd.jpg", cv2.IMREAD_GRAYSCALE)

img_kp = cv2.drawKeypoints(img, get_features(img)[0], None)

cv2.imshow("SURF_image", img_kp)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()