
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def mini_imagenet_folders():
    train_folder = '/home/grcwoods/WZP/Datas/miniImagenet/Train'
    test_folder = '/home/grcwoods/WZP/Datas/miniImagenet/Test'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders
def ImageNet_Dog_imagenet_folders():
    train_folder = '/home/grcwoods/WZP/Datas/ImageNet-Dog/train'
    test_folder = '/home/grcwoods/WZP/Datas/ImageNet-Dog/test'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders
def voc_imagenet_folders():
    train_folder = '/home/grcwoods/WZP/Datas/VOC2012/train'
    test_folder = '/home/grcwoods/WZP/Datas/VOC2012/test'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders
def cifar100_imagenet_folders():
    train_folder = '/home/grcwoods/WZP/Datas/cifar-100-python/Trainimages'
    test_folder = '/home/grcwoods/WZP/Datas/cifar-100-python/Testimages'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders
def omniglot_character_folders():
    data_folder ='/home/grcwoods/WZP/Datas/omniglot_resized'
    character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
    labels = np.array(range(len(character_folders))) 
    character_folders_labels = dict(zip(character_folders, labels))
    parent_character_folders = [os.path.join(data_folder, family) \
                for family in os.listdir(data_folder)]
    parent_labels = np.array(range(len(parent_character_folders)))
    parent_labels = dict(zip(parent_character_folders,parent_labels))
    parent_character_folders_labels = [parent_labels['/' + os.path.join(*c.split('/')[:-1])] for c in character_folders]
    # random.seed(1)
    random.shuffle(character_folders)
    num_train = 1200
    metatrain_character_folders = character_folders[:num_train]
    metaval_character_folders = character_folders[num_train:]
    return metatrain_character_folders,metaval_character_folders
def cifar10_imagenet_folders():
    train_folder = '/home/grcwoods/WZP/Datas/cifar-10-python/Train'
    test_folder = '/home/grcwoods/WZP/Datas/cifar-10-python/Test'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders
def toy_folders():
    train_folder = '/home/grcwoods/WZP/Datas/toy2/train'
    test_folder = '/home/grcwoods/WZP/Datas/toy2/test'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders
def cifarFC100_imagenet_folders():
    train_folder = '/home/grcwoods/WZP/Datas/cifarfc/train'
    test_folder = '/home/grcwoods/WZP/Datas/cifarfc/test'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders
def MNIST_imagenet_folders():
    train_folder = '/home/grcwoods/WZP/Datas/MNIST/Train'
    test_folder = '/home/grcwoods/WZP/Datas/MNIST/Test'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders
def STL10_imagenet_folders():
    train_folder = '/home/grcwoods/WZP/Datas/STL-10/Train'
    test_folder = '/home/grcwoods/WZP/Datas/STL-10/Test'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders

class MiniImagenetTask(object):
    def __init__(self, character_folders, num_classes, train_num,test_num):

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]

        self.train_labels = [labels['/'+self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels['/'+self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

class FewShotDataset(Dataset):
    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class MiniImagenet(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(MiniImagenet, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        #print(np.shape(image),np.shape(label))
        return image, label

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_cl, num_inst,shuffle=True):

        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        else:
            batches = [[i+j*self.num_inst for i in range(self.num_inst)] for j in range(self.num_cl)]
        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                   random.shuffle(sublist)
        batches = [item for sublist in batches for item in sublist]
        return iter(batches)

    def __len__(self):
        return 1

class ClassBalancedSamplerOld(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def get_mini_imagenet_data_loader(task, num_per_class=1, split='train',shuffle = True):
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    dataset = MiniImagenet(task,split=split,transform=transforms.Compose([transforms.Resize((84,84)),transforms.ToTensor(),normalize]))
    if split == 'train':
        sampler = ClassBalancedSamplerOld(num_per_class,task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(task.num_classes, task.test_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)
    return loader
