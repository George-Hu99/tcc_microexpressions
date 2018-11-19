import os
import operator
from functools import reduce
import numpy as np
from PIL import Image
import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import decomposition
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage import data, exposure

from sklearn.decomposition import PCA

class MicroExpression(object):
    def __init__(self, path, df, w=120, h=150):
        CASME_ans = {
            'happiness' : 0, 
            'disgust' : 1, 
            'repression' : 2, 
            'fear' : 3, 
            'sadness' : 4, 
            'surprise' : 5, 
            'others' : 6
        }

        CASME_to_SMIC = {
            'happiness' : 0, 
            'disgust' : 1, 
            'repression' : 1, 
            'fear' : 1, 
            'sadness' : 1, 
            'surprise' : 2, 
            'others' : 3
        }

        SMIC_ans = {
            'positive' : 0,
            'negative' : 1,
            'surprise' : 2,
            'default' : 3
        }
        self.path = path
        self.image_list = []
        self.filename = []
        for i, filename in enumerate(os.listdir(path)):
            im = Image.open('{path}/{filename}'.format(path=path, filename=filename))
            im = im.resize((w,h), Image.ANTIALIAS)
            im = np.array(im.convert('L'))
            self.image_list.append(im.astype('float'))
            self.filename.append(filename)

        self.qtd_images = len(self.image_list)

        self.dataset = "SMIC_all_cropped" if self.path.find("SMIC_all_cropped") != -1 else "Cropped"
        self.image = sum(self.image_list)/len(self.image_list)
        self.sum_images = (np.asarray([[[0] for i in range(120)] for j in range(150)]).reshape(150,120)).astype('float')
        for i in range(len(self.image_list)):
            self.sum_images += self.image_list[i]

        if self.dataset == "SMIC_all_cropped":
            self.subj = int(path.split('/')[9][1:])
            if self.path.find('HS') != -1:
                self.camera = 'HS'
            elif self.path.find('VIS') != -1:
                self.camera = 'VIS'
            elif self.path.find('NIR') != -1:
                self.camera = 'NIR'

            if self.path.find("micro") != -1:
                self.micro = 'micro'
            else: self.micro = 'non_micro'

            if self.path.find("positive") != -1:
                self.type = "positive"
                self.ans_bool = 1
            elif self.path.find("negative") != -1:
                self.type = "negative"                
                self.ans_bool = 1
            elif self.path.find("surprise") != -1:
                self.type = "surprise"
                self.ans_bool = 1
            else:
                self.type = "default"
                self.ans_bool = 0

            self.ans_smic = SMIC_ans[self.type]
            self.ans_casme = "other"
        
        else:
            self.camera = "HS"
            self.micro = 'micro'

            self.subj = int(self.path[self.path.find('sub')+3 : self.path.find('sub')+5])
            self.df = df
            name = self.path[self.path.find('EP') : len(self.path)]
            data = self.df[(self.df.Subject == self.subj) & (self.df.Filename == name)]
            self.type = data.EstimatedEmotion.values[0]
        
            self.ans_smic = CASME_to_SMIC[self.type]
            self.ans_casme = CASME_ans[self.type]
            self.ans_bool = 1

    def apply_image(self, matrix):
        self.result_image = abs(self.image - matrix)

    def apply_hog(self, orientations=9 , mean=0, pixels_per_cell=(10, 10),
	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
	visualize=False, _PCA=False, pca_lvl=.95):      
        if mean == 0:
            self.hog_array = list(map(lambda x: 
                hog(x, 
                    orientations=orientations, 
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, 
                    visualize=False
                ), self.image_list))
        else:
            self.hog_array = hog(self.result_image, 
                    orientations=orientations, 
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, 
                    visualize=False
                )
            
    def apply_pca(self):
        pca = decomposition.PCA(n_components=1)
        self.pca_array = pca.fit_transform(self.result_image)
    def apply_lbp(self):
        self.lbp = (local_binary_pattern(self.result_image, 10, 3, method='uniform')).flatten()
    def apply_reg(self):
        self.reg_image = (self.result_image).flatten()

    def summarize_hog(self, way=1):
        if way == 1:
            self.s_hog = reduce(operator.add, self.hog_array)
            self.s_hog = np.divide(self.s_hog, self.qtd_images)
