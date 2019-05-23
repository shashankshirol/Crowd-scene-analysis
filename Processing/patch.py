import cv2
import os

def split_img(img_loc, width, height):
    if not os.path.exists('patches'):
        os.makedirs('patches')
    img = cv2.imread(img_loc)
    for r in range(0, img.shape[0], width):
        for c in range(0, img.shape[1], height):
            cv2.imwrite("patches/"+img_loc[:img_loc.find('.')]+str(r)+"_"+str(c)+".png", img[r:r+width, c:c+height, :])


split_img("0310.jpg", 64, 64) #splits the image into 64x64 patches. For any other size, specify here.
