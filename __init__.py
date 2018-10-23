from Auxiliar.MakeDictByDir import MakeDictByDir
from Auxiliar.MicroExpression import MicroExpression
import numpy as np
import os
import pickle
import cv2
from sklearn.model_selection import train_test_split
from itertools import *
from sklearn.model_selection import cross_val_score
from sklearn import svm
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier

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

def make_equal(arrays, tam_max):
    for i, array in enumerate(arrays):
        if len(array) < tam_max:
            array = np.append(array, (tam_max-len(array)) * [0])
    return arrays
            
            

if __name__ == "__main__":
    dir = '/home/henriquehaji/PycharmProjects/TCC/PreProcess/Datasets/SMIC_all_cropped/'
    path_imgs = getImages(dir, list())
    #hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
    #derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,
    #nlevels, signedGradients)
    maior = 0
    X = []
    with open('hog_arrays.txt', 'a') as file:
        for i, path in enumerate(path_imgs):
            #print(i)
            micro = MicroExpression(path)
            micro.apply_hog(_PCA=False)
            micro.summarize_hog(1)

            #file.write(','.join(map(str,micro.s_hog)) + '\n')

            X.append(micro.s_hog)


            if len(X[i]) > maior:
                maior = len(X[i])
            #for j, array in enumerate(micro.hog_array):
                #file.write(','.join(map(str,micro.hog_array[j])) + '\n')
            #object_to_file((micro.hog_array,micro.type), 'hog_arrays/{}'.format(i))

    for i, array in enumerate(X):
        if len(array) < maior:
            X[i] = np.append(array, (maior-len(array)) * [0])

    y = []
    with open('answers.txt', 'a') as file:
        for i, path in enumerate(path_imgs):
            micro = MicroExpression(path)
            y.append(micro.ans)
            #file.write(''.join(micro.type) + '\n')
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #SVM
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    acertos = sum(clf.predict(X_test) == y_test)
    scores = cross_val_score(clf, X_test, y_test, cv=50)

    #Random Forest
    clf = RandomForestClassifier(n_jobs=2, random_state=0).fit(X_train, y_train)
    acertos = sum(clf.predict(X_test) == y_test)
    scores = cross_val_score(clf, X_test, y_test, cv=50)
    
    #Deep
    from keras.utils import to_categorical
    y_binary = to_categorical(y_train)

    model = Sequential()

    model.add(Dense(units=1, activation='relu', input_dim=11520))
    model.add(Dense(units=4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
    model.fit(np.array(X_train), np.array(y_binary), epochs=5, batch_size=32)
    

    

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.train(X_train, cv2.ml.ROW_SAMPLE,np.array(y_train))



    #object_to_file(microexpressions, 'PreProcess/hog_arrays/{micro}.txt'.format(micro=num))