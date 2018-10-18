from PIL import Image
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure


class MicroExpression(object):
    def __init__(self, path, ext='bmp'):
        self.path = '{}/*.{}'.format(path, ext)

        self.image_list = []
        for filename in glob.glob(self.path):
            #im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im = np.array(Image.open(filename))
            self.image_list.append(im)

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
            elif self.path.find("negative") != -1:
                self.type = "negative"
            elif self.path.find("surprise") != -1:
                self.type = "surprise"
            else:
                self.type = "default"

        else:
            self.camera = "no_cam"

        #hog = cv.HOGDescriptor()
        #self.hog_array = list(map(lambda x: hog.compute(x), self.image_list))

    def apply_hog(self, orientations=8, pixels_per_cell=(16,16), 
                cells_per_block=(1, 1),visualize=True,multichannel=True):
        self.hog_array = list(map(lambda x: 
            hog(x, 
                orientations=orientations, 
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block, 
                visualize=visualize, 
                multichannel=multichannel
            )[1], self.image_list))

