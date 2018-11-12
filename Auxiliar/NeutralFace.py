import numpy as np

class NeutralFace(object):
    def __init__(self):
        self.matrix = (np.asarray([[[0] for i in range(120)] for j in range(150)]).reshape(150,120)).astype('float')
        self.len = 0
        
    
    def add_mat(self, mat):
        self.matrix = self.matrix + mat
        self.len += 1
    
    def gen_neutral(self):
        self.neutral = self.matrix/self.len
        return self.neutral