import numpy as np
import pandas as pd
import os
import open_clip
import torch
from PIL import Image


model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',cache_dir='../clip')
train_images = []
test_images = []
train_labels = []
test_labels = []
from tqdm import tqdm
import torchvision
train_set = torchvision.datasets.Food101(root='/mnt/localssd/cifar',split='train',download=True,transform=test_preprocess)
test_set = torchvision.datasets.Food101(root='/mnt/localssd/cifar',split='test',download=True,transform=test_preprocess)
for (image,label) in tqdm(train_set):
    train_images.append(torch.unsqueeze(image,0))
    train_labels.append(label)
for (image,label) in tqdm(test_set):
    test_images.append(torch.unsqueeze(image,0))
    test_labels.append(label)
print(test_labels)
def get_few_shot_idx(train_y, K, classes):
    idx = []
    print(train_y.shape)
    for c in range(classes):
        id_c = np.argwhere(train_y==c)
        id_c = np.squeeze(id_c)
        idd = np.random.choice(id_c,K,replace=False)
        idx += list(idd)
    idx = np.array(idx)
    print(idx)
    return idx
train_images = torch.stack(train_images,dim=1)
train_images = torch.squeeze(train_images)
test_images = torch.stack(test_images,dim=1)
test_images = torch.squeeze(test_images)
print(train_images.shape)
print(test_images.shape)
index_1shot = get_few_shot_idx(np.array(train_labels),1,100)
index_2shot = get_few_shot_idx(np.array(train_labels),2,100)
index_4shot = get_few_shot_idx(np.array(train_labels),4,100)
index_8shot = get_few_shot_idx(np.array(train_labels),8,100)
index_16shot = get_few_shot_idx(np.array(train_labels),16,100)
train_labels = torch.LongTensor(train_labels)
test_labels = torch.LongTensor(test_labels)
train_image_1shot = train_images[index_1shot]
train_image_2shot = train_images[index_2shot]
train_image_4shot = train_images[index_4shot]
train_image_8shot = train_images[index_8shot]
train_image_16shot = train_images[index_16shot]
train_label_1shot = train_labels[index_1shot]
train_label_2shot = train_labels[index_2shot]
train_label_4shot = train_labels[index_4shot]
train_label_8shot = train_labels[index_8shot]
train_label_16shot = train_labels[index_16shot]
data_root = 'data/'
torch.save({'image':train_image_1shot,'label':train_label_1shot},data_root+'train_data_1shot.pt')
torch.save({'image':train_image_2shot,'label':train_label_2shot},data_root+'train_data_2shot.pt')
torch.save({'image':train_image_4shot,'label':train_label_4shot},data_root+'train_data_4shot.pt')
torch.save({'image':train_image_8shot,'label':train_label_8shot},data_root+'train_data_8shot.pt')
torch.save({'image':train_image_16shot,'label':train_label_16shot},data_root+'train_data_16shot.pt')
torch.save({'image':train_images,'label':train_labels},data_root+'train_data_allshot.pt')
torch.save({'image':test_images,'label':test_labels},data_root+'test_data.pt')
