import sklearn
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
'''
x1 = [0,0,0,1,0,1,0,0,0]
x2 = [0,0,0,0,0,0,0,0,1]
y = [0.1,0.2,0.1,0.8,0.75,0.62,0.33,0.19,0.17]
#print(mutual_info_regression([[t] for t in x1],y))
#print(mutual_info_score(x1,y))

'''
from tqdm import tqdm
import os
import numpy as np
import torch
import open_clip
from PIL import Image
from torch import nn
import pandas as pd
import json
import sys



device = 'cuda:0'


data_root = 'data/'
with torch.no_grad():

    data = torch.load(data_root+'train_data_allshot.pt')
    images = data['image']
    model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model.load_state_dict(torch.load('checkpoint_allshot/' + 'model400_99.pt',
                                     map_location='cuda:0'))
    model.to(device)
    def process_text(texts):
        device = 'cuda:0'
        with torch.no_grad():
            texts = tokenizer(texts)
            texts = texts.to(device)
            text_feature = model.encode_text(texts)
        return text_feature

    def process_image(imgs):
        device = 'cuda:0'
        with torch.no_grad():
            imgs = imgs.to(device)
            image_feature = model.encode_image(imgs)
        return image_feature

    def batchify_run(process_fn, data_lst, res, batch_size, use_tqdm=True):
        data_lst_len = len(data_lst)
        num_batch = np.ceil(data_lst_len / batch_size).astype(int)
        iterator = range(num_batch)
        if use_tqdm:
            iterator = tqdm(iterator)
        for i in iterator:
            batch_data = data_lst[i * batch_size:(i + 1) * batch_size]
            batch_res = process_fn(batch_data)
            res[i * batch_size:(i + 1) * batch_size] = batch_res
            del batch_res



    labels = data['label']
    with open('all_att_90.json','r') as fp:
        concepts=json.load(fp)
    class_label=np.load('class_label_des_90.npy')
    image_features = torch.load('img_embedding_all_400/'+'img_train_99.pt')
    text_features = torch.empty((len(concepts),768))
    batchify_run(process_text,concepts,text_features,512)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    print(image_features.shape)
    print(text_features.shape)
    mutual_info = []
    for i in tqdm(range(len(concepts))):
        x = []
        cand_ids = np.random.choice(len(labels),len(labels),replace=False)
        pos_ct=0
        neg_ct=0
        num_samples_pos = 30
        num_samples_neg = 30
        choose_id = []
        #print(cand_ids)
        #print(labels)
        for c_id in cand_ids:
            if class_label[labels[c_id]][i]==1:
                if pos_ct>=30:
                    continue
                choose_id.append(c_id)
                x.append([1])
                pos_ct+=1
        for c_id in cand_ids:
            if class_label[labels[c_id]][i] == 0:
                if neg_ct >= num_samples_neg:
                    continue
                choose_id.append(c_id)
                x.append([0])
                neg_ct += 1
        choose_id = np.array(choose_id)
        y = text_features[i]@image_features[choose_id].T
        #print(y.shape)
        y = y.squeeze(0)
        y = y.cpu().numpy()
        x = np.array(x)
        #print(x.shape)
        #print(y.shape)
        mutual_info.append(mutual_info_regression(x, y)[0])

ids = np.argsort(mutual_info)[::-1]
mutual_info = np.array(mutual_info)
print(0,mutual_info[ids[0]])
print(100,mutual_info[ids[100]])
print(500,mutual_info[ids[500]])
np.save('mutual_info_train.npy',mutual_info)


