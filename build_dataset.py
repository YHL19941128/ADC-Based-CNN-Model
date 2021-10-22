# -*- coding = utf-8 -*-

import torch
import os, glob
import random, csv
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class Mydata(Dataset):
    def __init__(self, root, resize, mode):
        super(Mydata, self).__init__()
        self.root = root
        self.resize = resize

        self.classifylabel = {}
        for name in sorted(os.listdir(root)):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            else:
                self.classifylabel[name] = len(self.classifylabel.keys())

        self.imagespath, self.labels = self.load_csv('images.csv')
        if mode=='train':
            self.imagespath = self.imagespath[:int(0.6 * len(self.imagespath))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode=='val':
            self.imagespath = self.imagespath[int(0.6 * len(self.imagespath)):int(0.8 * len(self.imagespath))]
            self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        else:
            self.imagespath = self.imagespath[int(0.8 * len(self.imagespath)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images_path_list = []
            for name in self.classifylabel.keys():
                images_path_list += glob.glob(os.path.join(self.root, name, '*.png'))
                images_path_list += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images_path_list += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            random.shuffle(images_path_list)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for imgpath in images_path_list:
                    name = imgpath.split(os.sep)[-2]
                    label = self.classifylabel[name]
                    writer.writerow([imgpath, label])
                print('writen into csv file:', filename)
        else:
            imagespath, labels = [], []
            with open(os.path.join(self.root, filename)) as f:
                reader = csv.reader(f)
                for row in reader:
                    # '.\\mask_dataset\\image_nomask\\0650.jpg', 0
                    imgpath, strlabel = row
                    label = int(strlabel)
                    imagespath.append(imgpath)
                    labels.append(label)
        return imagespath, labels

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x_hat = (x-mean)/std
        # x = x_hat*std + mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __len__(self):
        return len(self.imagespath)

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.imagespath, self.labels
        # imgpath: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0

        imgpath, label = self.imagespath[idx], self.labels[idx]
        imgdata_from_path = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        imgtensor = imgdata_from_path(imgpath)
        labeltensor = torch.tensor(label)

        return imgtensor, labeltensor ,imgpath



