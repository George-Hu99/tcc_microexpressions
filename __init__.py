from Auxiliar.MakeDictByDir import MakeDictByDir
from Auxiliar.MicroExpression import MicroExpression
import numpy as np
import os
import pickle
import cv2

# Muda o tamanho de várias imagens abaixo de um diretório
def rescale(dir):
    for filename in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, filename)):
            rescale(os.path.join(dir, filename))
        elif filename.endswith(".jpg") and os.listdir(dir)[0].endswith(".jpg"):
            os.system("mogrify -resize 131x161 -quality 100 -format bmp  {0}/*.jpg".format(dir))
            os.system("find {0} -name '*.jpg' -delete".format(dir))

# Gera uma string para cada pasta de microexpressão
def getImages(diretorio, path_imgs):
    for filename in os.listdir(diretorio):
        if os.path.isdir(os.path.join(diretorio, filename)):
            getImages(os.path.join(diretorio, filename), path_imgs)
        elif filename.endswith('.bmp'):
            path_imgs.append(diretorio)
            break
    return path_imgs

def par_map_func(items, func):
    pool = mp.Pool(processes=4)
    return pool.map(func, items)

def object_to_file(object, filename):
    with open(filename, "wb") as fp:
        pickle.dump(object, fp)

def file_to_object(filename):
    with open(filename, "rb") as fp:
        return pickle.load(fp)


if __name__ == "__main__":
    dir = '/home/henriquehaji/PycharmProjects/TCC/PreProcess/Datasets/SMIC_all_cropped/'
    path_imgs = getImages(dir, list())
    #hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
    #derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,
    #nlevels, signedGradients)
    X = []
    with open('hog_arrays.txt', 'a') as file:
        for i, path in enumerate(path_imgs):
            #print(i)
            micro = MicroExpression(path)
            micro.apply_hog(_PCA=False)
            micro.summarize_hog(1)
            #file.write(','.join(map(str,micro.s_hog)) + '\n')
            X.append(micro.s_hog)
            #for j, array in enumerate(micro.hog_array):
                #file.write(','.join(map(str,micro.hog_array[j])) + '\n')
            #object_to_file((micro.hog_array,micro.type), 'hog_arrays/{}'.format(i))

    

    y = []
    with open('answers.txt', 'a') as file:
        for i, path in enumerate(path_imgs):
            micro = MicroExpression(path)
            y.append(micro.ans)
            #file.write(''.join(micro.type) + '\n')
    trainLabels = np.array(y_train, dtype=int)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.train(X_train, cv2.ml.ROW_SAMPLE,np.array(y_train))



    #object_to_file(microexpressions, 'PreProcess/hog_arrays/{micro}.txt'.format(micro=num))