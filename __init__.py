import os
import cv2
import pickle
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Flatten
import keras
from itertools import *
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from datetime import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from Auxiliar.MakeDictByDir import MakeDictByDir
from Auxiliar.MicroExpression import MicroExpression
from Auxiliar.NeutralFace import NeutralFace
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
 
def createModel(input_shape, nClasses):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses))
    model.add(Activation('softmax'))
    return model


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


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

def gen_neutral_files(path_imgs, ans_df):
    nf = [NeutralFace() for i in range(57)]
    for i, path in enumerate(path_imgs):
        micro = MicroExpression(path, df=ans_df)
        if (path.find('Cropped') > 0):
            nf[micro.subj-1].add_mat(micro.sum_images)
            nf[micro.subj-1].len += micro.qtd_images
        elif (path.find('SMIC_all_cropped/HS') > 0):
            nf[micro.subj+25].add_mat(micro.sum_images)
            nf[micro.subj+25].len += micro.qtd_images
        '''
        elif (path.find('SMIC_all_cropped/VIS') > 0):
            nf[micro.subj+31].add_mat(micro.sum_images)
            nf[micro.subj+31].len += micro.qtd_images
        elif (path.find('SMIC_all_cropped/NIR') > 0):
            nf[micro.subj+39].add_mat(micro.sum_images)
            nf[micro.subj+39].len += micro.qtd_images
        '''
    for i in range(57):
        nf[i].gen_neutral()
        np.savetxt('neutral_faces/' + str(i), nf[i].neutral, fmt='%0.f')

def get_neutral_faces():
    path = '/home/henriquehaji/tcc_microexpressions/neutral_faces'
    faces = [0] * 42
    for filename in os.listdir(path):
        faces[int(filename)] = np.loadtxt(path + '/' + str(filename))
    return faces

def get_subj(path, subj):
    if (path.find('Cropped') > 0):
        return micro.subj-1
    elif (path.find('SMIC_all_cropped/HS') > 0):
        return micro.subj+25
    '''        
    elif (path.find('SMIC_all_cropped/VIS') > 0):
        return micro.subj+45-10
    elif (path.find('SMIC_all_cropped/NIR') > 0):
        return micro.subj+53-10
    '''


if __name__ == "__main__":
    dir_smic = '/home/henriquehaji/PycharmProjects/TCC/PreProcess/Datasets/SMIC_all_cropped/HS'
    dir_norm = '/home/henriquehaji/PycharmProjects/TCC/PreProcess/Datasets/Cropped'
    
    path_imgs = getImages(dir_smic, list())
    path_imgs = getImages(dir_norm, path_imgs)
    neutral_images = get_neutral_faces()

    ans_df = pd.read_csv('CASME2_Ans.csv', ',')
    #hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
    #derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,
    #nlevels, signedGradients)
    
    X_hog = []
    X_pca = []
    X_image = []
    X_lbp = []
    X_reg = []
    X_tensor = []
    for i, path in enumerate(path_imgs):
        micro = MicroExpression(path, ans_df)
        micro.apply_image(neutral_images[get_subj(micro.path, micro.subj)])
        micro.apply_pca()
        X_pca.append(micro.pca_array)
        #X_tensor.append(micro.result_image)
        #micro.apply_hog(mean=1)
        #micro.apply_lbp()
        #micro.apply_reg()
        #micro.summarize_hog(1)
        #X = np.append(X, micro.s_hog)
        #X_hog.append(micro.hog_array)
        #X_lbp.append(micro.lbp)
        #X_reg.append(micro.reg_image)
        #X_reg.append(micro.result_image)
    
    np.savetxt('mean_hog.txt', X_hog, fmt='%.4f')
    np.savetxt('mean_lbp.txt', X_lbp, fmt='%.4f')
    np.savetxt('mean_reg.txt', X_reg, fmt='%.4f')
    np.savetxt('mean_pca.txt', X_pca, fmt='%.4f')

    X_hog = np.loadtxt('mean_hog.txt')
    X_lbp = np.loadtxt('mean_lbp.txt')
    X_hog = np.loadtxt('mean_reg.txt')
    X_pca = np.loadtxt('mean_pca.txt')

    y_bool = np.loadtxt('ans_bool.txt')
    y_smic = np.loadtxt('ans_smic.txt')
    y_bool = np.array([])
    y_smic = np.array([])
    #y_casme = np.array([])
    for i, path in enumerate(path_imgs):
        micro = MicroExpression(path, ans_df)
        y_smic = np.append(y_smic, micro.ans_smic)
    np.savetxt('ans_smic.txt', y_smic, fmt='%.4f')
    for i, path in enumerate(path_imgs):
        micro = MicroExpression(path, ans_df)
        y_bool = np.append(y_bool, micro.ans_bool)
    np.savetxt('ans_bool.txt', y_bool, fmt='%.4f')

    
    #kf = sklearn.model_selection.KFold(30, shuffle=True)
    '''
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf_svm_l = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        score_svm_l = np.append(score_svm_l, sum(clf_svm_l.predict(X_test) == y_test))
    '''
    t_start = dt.now()

    y_pred = []
    y_ans = []
    Y_test =  np.array([])
    score = np.array([])
    kick =  np.array([])
    y_score = np.array([])
    X = np.asarray(X_pca)

    X_shuffle, y_shuffle = shuffle(X, y_smic, random_state=0)

    X_lou = X_shuffle[0:58, :]
    y_lou = y_shuffle[0:58]
    
    X_tt = X_shuffle[58:, :]
    y_tt = y_shuffle[58:]

    
    classifier = svm.SVC(kernel='linear', C=1, gamma=1)
    for i in range(100):
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X_tt):
            classifier.fit(X_tt[train_index], y_tt[train_index])
            #classifier = RandomForestClassifier(n_jobs=3, n_estimators=100).fit(X[train_index], y_smic[train_index])
            #classifier = GaussianNB().fit(X[train_index], y_smic[train_index])
            #y_score = np.vstack((y_score, classifier.decision_function(X[test_index]))) if len(y_score) != 0 else classifier.decision_function(X[test_index])
            #y_score = np.vstack((y_score, classifier.predict_proba(X[test_index]))) if len(y_score) != 0 else classifier.predict_proba(X[test_index])
            #y_score = np.append(y_score, classifier.predict_proba(X[test_index]))
            score = np.append(score, (sum(classifier.predict(X[test_index]) == y_tt[test_index])))
        
        y_pred = np.append(y_pred, classifier.predict(X[test_index]))
        y_ans = np.append(y_ans, y_smic[test_index])
        Y_test = np.append(Y_test, y_smic[test_index])
        
'''
    Binário
        Hog
            SVM-Linear - 79,08% Fotos: x
            Naive - 72,25% Fotos: x
            RF - 73,14% Fotos: x
        PCA
            SVM - 73,29%
            Naive - 67,61%
            RF - 73,22%
        Vazio
            DNN 69,17%/Loss 61,76% (1|40 min) - 70,54%/Loss 49.75% (2|15 min) - 80,13%/Loss 1,32% (3|5 min com 20 epocas) 89% com 100 Epocas
    SMIC
        Hog
            SVM-Linear - 53.47%
            Naive - 45.82%
            RF - 53,32%
        PCA
            SVM 45,77%
            Naive 29,05%
            RF 45,62%
        Vazio
            DNN 56%
'''
    total_hist = {}
    total_hist['acc'] = []
    total_hist['loss'] = []

    test_loss_arr = []
    test_acc_arr = []

    X_shuffle, y_shuffle = shuffle(X_tensor, y_smic_c, random_state=0)

    X_tensor_lou = X_shuffle[0:58, :, :, :]
    X_tensor_tt = X_shuffle[58:, :, :, :]

    y_smic_lou = y_shuffle[0:58, :]
    y_smic_tt = y_shuffle[58:, :]

    kf = KFold(n_splits=10, shuffle=True)
    input_shape = (150, 120, 1)
    nClasses = 4
    model = createModel(input_shape, nClasses)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    for train_index, test_index in kf.split(X_tensor_tt):
        #classifier = svm.SVC(kernel='linear', C=1, gamma=1).fit(X[train_index], y_smic[train_index])
        history = model.fit(X_tensor_tt[train_index], y_smic_tt[train_index], epochs=10, batch_size=16)
        total_hist['acc'].append(history.history['acc'])
        total_hist['loss'].append(history.history['loss'])

        test_loss, test_acc = model.evaluate(X_tensor_tt[test_index], y_smic_tt[test_index])

        test_loss_arr.append(test_loss)
        test_acc_arr.append(test_acc)
    
    test_loss, test_acc = model.evaluate(X_tensor_lou, y_smic_lou)
    test_loss_arr.append(test_loss)
    test_acc_arr.append(test_acc)

    y_pred = (model.predict_classes(X_tensor_lou, verbose=1))
    y_ans = np.argmax(y_smic_lou, axis=1)

    
    t_end = dt.now()
    print(t_end - t_start)
    
    #Deep
    X_train, X_test, y_train, y_test = train_test_split(X_tensor_norm, y_bool_norm, test_size=0.25, random_state=42)
    #y_binary = to_categorical(y_train)
    model.compile(loss='binary_crossentropy',
                optimizer=sgd,
                metrics=['binary_accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss=keras.losses.sparse_categorical_crossentropy,
    #            optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
    model.fit(X_train, y_train, epochs=50, batch_size=16)
    test_loss, test_acc = model.evaluate(X_test, y_test)

    num_classes = 2
    model = keras.Sequential()
    model.add(ZeroPadding1D(2, input_shape=(1,5544)))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=(5), strides=(1),
                    activation='relu'))
    #model.add(MaxPooling1D(pool_size=(2), strides=(1)))
    model.add(Conv1D(64, (5), activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.SGD(lr=0.01),
                metrics=['binary_accuracy'])
    model.fit(list(X_train), np.array(y_train), epochs=5)

    model = keras.Sequential()
    adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.add(Dense(1, input_dim=5544))
    model.add(Dense(2774, activation='tanh'))
    #model.add(Dense(1386, activation='softplus'))
    #model.add(Dense(693, activation='softsign'))
    #model.add(Dense(346, activation='tanh'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                optimizer=sgd,
                metrics=['binary_accuracy'])
    #model.compile(loss=keras.losses.sparse_categorical_crossentropy,
    #            optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
    model.fit(X_train, np.array(y_train), epochs=5, batch_size=64)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)



    #object_to_file(microexpressions, 'PreProcess/hog_arrays/{micro}.txt'.format(micro=num))