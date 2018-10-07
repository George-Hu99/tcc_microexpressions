from PreProcess.Auxiliar.MicroExpression import MicroExpression
from PreProcess.Auxiliar.MakeDictByDir import MakeDictByDir
import os


class MicroExpressionRecord(object):
    def __init__(self, micro, subject, cam, type, dataset, ext='bmp'):
        self.micro = micro
        self.subject = subject
        self.cam = cam
        self.type = type
        self.dataset = dataset
        self.ext = ext
        self.micro_exps = self.listImages()

    def listImages(self):
        self.dir = '/home/henriquehaji/PycharmProjects/TCC/PreProcess/Datasets/{dataset}/{cam}/{subject}/{micro}/'
        if self.micro == 'micro':
            self.dir = self.dir + '{type}/'
            self.dir = self.dir.format(
                dataset=self.dataset, cam=self.cam, subject=self.subject, micro=self.micro, type=self.type
            )
            dic = MakeDictByDir.get_directory_structure(self.dir)
            return list(map(lambda x: MicroExpression(self.dir + '/' + x, self.ext), dic[self.type]))
        else:
            self.dir.format(
                dataset=self.dataset, cam=self.cam, subject=self.subject, micro=self.micro
            )
            return list(MicroExpression(self.dir, self.ext))

    def getImages(self, new_dir):
        for filename in os.listdir(dir):
            if os.path.isdir(os.path.join(dir, filename)):
                self.getImages(os.path.join(new_dir, filename))
            elif filename.endswith(self.ext):
                self.micro_exps = self.listImages()