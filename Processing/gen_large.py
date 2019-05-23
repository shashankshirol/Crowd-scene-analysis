import cv2
import os

def split_img(width, height):
    if not os.path.exists('patches'):
        os.makedirs('patches')
    images = os.listdir('data')
    
    for im in images:
        if not os.path.exists('patches/'+im[:im.find('.')]):
            os.makedirs('patches/'+im[:im.find('.')])
        img = cv2.imread(os.path.join('data', im))
        for r in range(0, img.shape[0], width):
            for c in range(0, img.shape[1], height):
                cv2.imwrite("patches/"+im[:im.find('.')]+"/"+im[:im.find('.')]+str(r)+"_"+str(c)+".png", img[r:r+width, c:c+height, :])

split_img(64, 64) #splits the images in a given directory into 64x64 patches. For any other size, specify here.
		  #Assumes images stored as: data -> images
