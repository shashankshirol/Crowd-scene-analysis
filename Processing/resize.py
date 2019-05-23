import os
import cv2

#Make Sure your directory doesn't have any other type of files. And only contains images.

def resize_imgs(path, width, height):
    if not os.path.exists('resize'):
        os.makedirs('resize')
    images = os.listdir(path)
    
    for i in range(len(images)):
        img = cv2.imread(os.path.join(path, images[i]), cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(img, (width, height))
        cv2.imwrite("resize/"+images[i], resized)
    
resize_imgs('background', 64, 64)