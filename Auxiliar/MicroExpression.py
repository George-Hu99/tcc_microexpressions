import os
import operator
from functools import reduce
import numpy as np
from PIL import Image
import cv2
import glob
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

from sklearn.decomposition import PCA

class MicroExpression(object):
    def __init__(self, path):
        self.path = path
        self.image_list = []
        self.filename = []
        for i, filename in enumerate(os.listdir(path)):
            im = np.array(Image.open('{path}/{filename}'.format(path=path, filename=filename)).convert('L'))
            self.image_list.append(im)
            self.filename.append(filename)

        self.qtd_images = len(self.image_list)

        self.dataset = "SMIC_all_cropped" if self.path.find("SMIC_all_cropped") != -1 else "Cropped_norm"

        if self.dataset == "SMIC_all_cropped":
            if self.path.find("HS") != -1:
                self.camera = "HS"
            elif self.path.find("VIS") != -1:
                self.camera = "VIS"
            self.camera = "NIR"

            if self.path.find("micro") != -1:
                self.micro = 'micro'
            else: self.micro = 'non_micro'

            if self.path.find("positive") != -1:
                self.type = "positive"
                self.ans = 3
            elif self.path.find("negative") != -1:
                self.type = "negative"
                self.ans = 2
            elif self.path.find("surprise") != -1:
                self.type = "surprise"
                self.ans = 1
            else:
                self.type = "default"
                self.ans = 0

        else:
            self.camera = "no_cam"
            self.type = "default"
            self.ans = 0

        #hog = cv.HOGDescriptor()
        #self.hog_array = list(map(lambda x: hog.compute(x), self.image_list))

    def apply_hog(self, orientations=9, pixels_per_cell=(10, 10),
	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
	visualize=False, _PCA=False, pca_lvl=.95):      
        #self.hog_array = list(map(lambda x: hog.compute(x), self.image_list))
        
        self.hog_array = list(map(lambda x: 
            hog(x, 
                orientations=orientations, 
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block, 
                visualize=False
            ), self.image_list))
        
        if _PCA == True:
            pca = PCA(pca_lvl)
            for index,hog_image in enumerate(self.hog_array):
                pca.fit(hog_image)
                self.hog_array[index] = pca.transform(hog_image)
    
    def summarize_hog(self, way=1):
        if way == 1:
            self.s_hog = reduce(operator.add, self.hog_array)
            self.s_hog = np.divide(self.s_hog, self.qtd_images)
            
        