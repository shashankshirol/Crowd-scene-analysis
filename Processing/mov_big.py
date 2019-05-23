import os
import shutil

path = 'patches'

dest = 'train'

img_folders = os.listdir(path)

for im_fold in img_folders:
    img_patches = os.listdir(os.path.join(path, im_fold))
    
    img_paths = [os.path.join(path, im_fold, i) for i in img_patches]
    img_patches = sorted(img_paths, key = os.path.getsize)


    largest_file = img_patches[-1] #Assumes the crowd patch of the image will have highest size owing to its color variations

    print(largest_file)
    shutil.move(largest_file, os.path.join(dest, largest_file[largest_file.rfind('/')+1:]))
    print('move complete!')
