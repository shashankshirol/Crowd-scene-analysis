import cv2
import numpy as np
from BOV import *
from sklearn.exceptions import DataConversionWarning
import warnings
import os

class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = 'data/training'
        self.surf = cv2.xfeatures2d.SURF_create()
        self.bov_helper = BOVHelpers(no_clusters)
        self.des_list = []

    def train_model(self):
        train_crowd = self.train_path + '/crowd'
        crowd_images = os.listdir(train_crowd)
        num_images_crowd = len(crowd_images)

            #Training Crowd Images
        for i in range(num_images_crowd):
            img = cv2.imread(os.path.join(train_crowd, crowd_images[i]), cv2.IMREAD_GRAYSCALE)
            _, des = self.surf.detectAndCompute(img, None)
            self.des_list.append(des)
            
        #-----------------------------------------------------------------------------------------------#

        #performing clustering

        bov_des_stack = self.bov_helper.formatND(self.des_list)
        self.bov_helper.cluster()
        self.bov_helper.developVocabulary(n_images = (num_images_crowd), descriptor_list = self.des_list)

        self.bov_helper.standardize()
        self.bov_helper.train()
    
    def recog(self, test_img, path = None):
        _, des = self.surf.detectAndCompute(test_img, None)

        vocab = np.array( [[0 for i in range(self.no_clusters)]])

        test_ret = self.bov_helper.kmeans_obj.predict(des)

        for each in test_ret:
            vocab[0][each] += 1

        vocab = self.bov_helper.scale.transform(vocab)

        result_label = self.bov_helper.clf.predict(vocab)
        
        return result_label

if __name__ == '__main__':
    bov = BOV(no_clusters = 125)
    bov.train_model()

    warnings.filterwarnings(action = 'ignore', category=DataConversionWarning)
    img = cv2.imread('test1.png', cv2.IMREAD_GRAYSCALE)
    print(bov.recog(img))

    img = cv2.imread('test2.png', cv2.IMREAD_GRAYSCALE)
    print(bov.recog(img))

    img = cv2.imread('test3.png', cv2.IMREAD_GRAYSCALE)
    print(bov.recog(img))