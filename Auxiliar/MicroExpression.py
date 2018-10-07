from PIL import Image
import numpy as np
import cv2 as cv
import glob


class MicroExpression(object):
    def __init__(self, path, ext='bmp'):
        self.path = '{}/*.{}'.format(path, ext)

        self.image_list = []
        for filename in glob.glob(self.path):
            im = cv.imread(filename)
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
            self.type = "surprise"

        else:
            self.camera = "no_cam"

        #hog = cv.HOGDescriptor()
        #self.hog_array = list(map(lambda x: hog.compute(x), self.image_list))

    def apply_hog(self, hog):

        self.hog_array = list(map(lambda x: hog.compute(x), self.image_list))

