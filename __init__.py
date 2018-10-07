from PreProcess.Auxiliar.MakeDictByDir import MakeDictByDir
from PreProcess.Auxiliar.MicroExpression import MicroExpression
import os
import pickle
import cv2 as cv

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
    dir = '/home/henriquehaji/PycharmProjects/TCC/PreProcess/Datasets'

    path_imgs = getImages(dir, list())

    array = 'PreProcess/hog_arrays/{}.txt'
    i = 0
    hog = cv.HOGDescriptor()
    for path in path_imgs:
        micro = MicroExpression(path)
        micro.apply_hog(hog)
        object_to_file(micro, 'PreProcess/hog_arrays/{}.txt'.format(i))
        i += 1



    #object_to_file(microexpressions, 'PreProcess/hog_arrays/{micro}.txt'.format(micro=num))