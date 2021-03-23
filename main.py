"""
Class Activation Mapping
Googlenet, Kaggle data
"""

from update import *
from data import *
from train import *
import torch, os
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from inception import inception_v3
from PIL import Image

from vgg16 import *

import random


# functions
CAM             = 1
USE_CUDA        = 1
RESUME          = 0
PRETRAINED      = 0


# hyperparameters
BATCH_SIZE      = 32
IMG_SIZE        = 224
LEARNING_RATE   = 0.002
EPOCH           = 2

#build datasets
DIR_TRAIN = "E:\CAM\train"
DIR_TEST = "E:\CAM\test"
class_to_int={'dog':0,'cat':1}




class catdogDataset(Dataset):
	def __init__(self,imgs,class_to_int,transform=None):
		super(Dataset,self).__init__()
    
		self.imgs=imgs
    
		self.class_to_int=class_to_int
		self.transform=transform
		
	def __getitem__(self,idx):
		img_name=self.imgs[idx]
		img=Image.open(DIR_TRAIN+img_name)
		img=self.transform(img)
		
    
		label=class_to_int[img_name.split('.')[0]]
		label=torch.tensor(label,dtype=torch.long)
		return img,label
    	
	def __len__(self):
		return len(self.imgs)


# prepare data
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

imgs=os.listdir(DIR_TRAIN)
random.shuffle(imgs)
train_data=catdogDataset(imgs[:int(len(imgs)*0.9)],class_to_int,transform_train)
test_data=catdogDataset(imgs[int(len(imgs)*0.9):],class_to_int,transform_test)

#train_data = datasets.ImageFolder('kaggle/working/train/', transform=transform_train)
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#test_data = datasets.ImageFolder('kaggle/working/test1/', transform=transform_test)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# class
classes = {0: 'dog', 1: 'cat'}


# fine tuning
if PRETRAINED:
    net = VGG(pretrained=False)
    net.load_state_dict(torch.load('/content/drive/MyDrive/inceptionv3.pth'))   #input your pretrained net
    print('load model successful\n')
    for param in net.parameters():
        param.requires_grad = False
    net.fc = torch.nn.Linear(2048, 2)
else:
    net = VGG(num_classes=2)


net.cuda()


# load checkpoint
if RESUME != 0:
    print("===> Resuming from checkpoint.")
    assert os.path.isfile('/content/drive/MyDrive/checkpoint/' + 'cam_inception' + str(RESUME) + '.pth'), 'Error: no checkpoint found!'
    net.load_state_dict(torch.load('checkpoint/' + str(RESUME) + '.pth'))


# retrain
criterion = torch.nn.CrossEntropyLoss()

if PRETRAINED:
    optimizer = torch.optim.SGD(net.fc.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

for epoch in range (1, EPOCH + 1):
    retrain(trainloader, net, USE_CUDA, epoch, criterion, optimizer)
    retest(testloader, net, USE_CUDA, criterion, epoch, RESUME)

