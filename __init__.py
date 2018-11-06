import os
import cv2
import pickle
import numpy as np
from itertools import *
from sklearn import svm
from datetime import datetime as dt
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import Kfold
from Auxiliar.MakeDictByDir import MakeDictByDir
from Auxiliar.MicroExpression import MicroExpression



from sklearn.model_selection import train_test_split


# Muda o tamanho de várias imagens abaixo de um diretório
def rescale(dir):
    for filename in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, filename)):
            rescale(os.path.join(dir, filename))
        elif filename.endswith(".jpg") and os.listdir(dir)[0].endswith(".jpg"):
            os.system("mogrify -quality 100 -format bmp {0}/*.jpg".format(dir))
            os.system("find {0} -name '*.jpg' -delete".format(dir))

# Gera uma string para cada pasta de microexpressão
def getImages(diretorio, path_imgs):
    for filename in os.listdir(diretorio):
        if os.path.isdir(os.path.join(diretorio, filename)):
            getImages(os.path.join(diretorio, filename), path_imgs)
        elif filename.endswith('.bmp') or filename.endswith('.jpg'):
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
    dir_smic = '/home/henriquehaji/PycharmProjects/TCC/PreProcess/Datasets/SMIC_all_cropped/HS'
    dir_norm = '/home/henriquehaji/PycharmProjects/TCC/PreProcess/Datasets/Cropped'
    
    path_imgs = getImages(dir_smic, list())
    path_imgs = getImages(dir_norm, path_imgs)

    #hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
    #derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,
    #nlevels, signedGradients)
    X = []
    with open('hog_arrays.txt', 'a') as file:
        for i, path in enumerate(path_imgs):
            micro = MicroExpression(path)
            micro.apply_hog(_PCA=False)
            micro.summarize_hog(1)
            #X = np.append(X, micro.s_hog)
            X.append(micro.s_hog)

    X = np.array(X)
    y = np.array([])
    with open('answers_bool.txt', 'a') as file:
        for i, path in enumerate(path_imgs):
            micro = MicroExpression(path)
            y = np.append(y, micro.ans_bool) 
    
    #kf = sklearn.model_selection.KFold(30, shuffle=True)
    '''
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf_svm_l = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        score_svm_l = np.append(score_svm_l, sum(clf_svm_l.predict(X_test) == y_test))
    '''
    t_start = dt.now()

    score_svm_l = np.array([])
    score_svm_rbf = np.array([])
    score_rf = np.array([])

    score_svm_l2 = np.array([])
    score_svm_rbf2 = np.array([])
    score_rf2 = np.array([])

    score_svm_l3 = np.array([])
    score_svm_rbf3 = np.array([])
    score_rf3 = np.array([])

    score_svm_l4 = np.array([])
    score_svm_rbf4 = np.array([])
    score_rf4 = np.array([])

    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        #SVM Linear
        clf_svm_l = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        score_svm_l = np.append(score_svm_l, sum(clf_svm_l.predict(X_test) == y_test))

        #SVM RBF
        clf_svm_rbf = svm.SVC(kernel='rbf', C=1, gamma='scale').fit(X_train, y_train)
        score_svm_rbf = np.append(score_svm_rbf, sum(clf_svm_rbf.predict(X_test) == y_test))

        #Random Forest
        clf_rf = RandomForestClassifier(n_jobs=3, random_state=0, n_estimators=100).fit(X_train, y_train)
        score_rf = np.append(score_rf, sum(clf_rf.predict(X_test) == y_test))

    for i in range(100):
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2)
        #SVM Linear
        clf_svm_l2 = svm.SVC(kernel='linear', C=1).fit(X_train2, y_train2)
        score_svm_l2 = np.append(score_svm_l2, sum(clf_svm_l2.predict(X_test2) == y_test2))

        #SVM RBF
        clf_svm_rbf2 = svm.SVC(kernel='rbf', C=1, gamma='scale').fit(X_train2, y_train2)
        score_svm_rbf2 = np.append(score_svm_rbf2, sum(clf_svm_rbf2.predict(X_test2) == y_test2))

        #Random Forest
        clf_rf2 = RandomForestClassifier(n_jobs=3, random_state=0, n_estimators=100).fit(X_train2, y_train2)
        score_rf2 = np.append(score_rf2, sum(clf_rf2.predict(X_test2) == y_test2))

        
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.3)
        #SVM Linear
        clf_svm_l3 = svm.SVC(kernel='linear', C=1).fit(X_train3, y_train3)
        score_svm_l3 = np.append(score_svm_l3, sum(clf_svm_l3.predict(X_test3) == y_test3))

        #SVM RBF
        clf_svm_rbf3 = svm.SVC(kernel='rbf', C=1, gamma='scale').fit(X_train3, y_train3)
        score_svm_rbf3 = np.append(score_svm_rbf3, sum(clf_svm_rbf3.predict(X_test3) == y_test3))

        #Random Forest
        clf_rf3 = RandomForestClassifier(n_jobs=3, random_state=0, n_estimators=100).fit(X_train3, y_train3)
        score_rf3 = np.append(score_rf3, sum(clf_rf3.predict(X_test3) == y_test3))

        X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y, test_size=0.4)
        #SVM Linear
        clf_svm_l4 = svm.SVC(kernel='linear', C=1).fit(X_train4, y_train4)
        score_svm_l4 = np.append(score_svm_l4, sum(clf_svm_l4.predict(X_test4) == y_test4))

        #SVM RBF
        clf_svm_rbf4 = svm.SVC(kernel='rbf', C=1, gamma='scale').fit(X_train4, y_train4)
        score_svm_rbf4 = np.append(score_svm_rbf4, sum(clf_svm_rbf4.predict(X_test4) == y_test4))

        #Random Forest
        clf_rf4 = RandomForestClassifier(n_jobs=4, random_state=0, n_estimators=100).fit(X_train4, y_train4)
        score_rf4 = np.append(score_rf4, sum(clf_rf4.predict(X_test4) == y_test4))
    


    
    t_end = dt.now()
    print(t_end - t_start)
    
    #Deep
    from keras.utils import to_categorical
    from keras.layers import Dense
    from keras.layers import Flatten
    import keras
    #y_binary = to_categorical(y_train)

    model = keras.Sequential()

    model.add(Dense(1, input_dim=5544))
    model.add(Dense(2772, activation='relu'))
    model.add(Dense(1386, activation='softplus'))
    model.add(Dense(693, activation='softsign'))
    model.add(Dense(346, activation='tanh'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    #model.compile(loss=keras.losses.sparse_categorical_crossentropy,
    #            optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
    model.fit(X_train4, np.array(y_train4), epochs=5, batch_size=32)
    test_loss, test_acc = model.evaluate(X_test4, y_test4)
    print('Test accuracy:', test_acc)



    #object_to_file(microexpressions, 'PreProcess/hog_arrays/{micro}.txt'.format(micro=num))