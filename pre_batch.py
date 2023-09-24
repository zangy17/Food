import argparse
import json
import os
import logging

import pandas as pd
import numpy as np
import torch
import open_clip
from PIL import Image


from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

from torch import nn

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

def main(shot,num_concept):

    # Load data

    data_root = 'data/'

    model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    #model.load_state_dict(torch.load('checkpoint/cub_des_x2o79.pt', map_location='cuda:0'))
    model.to(device)
    # Precompute attribute activations

    attribute_activations_train, attribute_activations_valid = [], []

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text)
    model_image = torch.nn.DataParallel(model_image)

    def process_text(texts):
        device = 'cuda:0'
        with torch.no_grad():
            texts = tokenizer(texts)
            texts = texts.to(device)
            text_feature = model_text(texts)
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
    with open('all_att_90.json','r') as fp:
        all_concepts=json.load(fp)
    if not os.path.exists('txt_embedding/' + 'all.pt'):
        all_text_features = torch.empty((len(all_concepts), 768))
        batchify_run(process_text, all_concepts, all_text_features, 1024)
    with open('select_concept_'+shot+'shot'+str(num_concept)+'.json', 'r') as fp:
        concepts = json.load(fp)


    train_images = torch.load(data_root+'train_data_'+shot+'shot.pt')['image']
    test_images = torch.load(data_root+'test_data.pt')['image']
    if os.path.exists('img_embedding/train_'+shot+'shot.pt'):
        train_image_features = torch.load('img_embedding/train_'+shot+'shot.pt')
    else:
        train_image_features = torch.empty((len(train_images),768))
        batchify_run(process_image, train_images, train_image_features, 1024)

        torch.save(train_image_features,'img_embedding/train_'+shot+'shot.pt')
    if os.path.exists('img_embedding/test.pt'):
        test_image_features = torch.load('img_embedding/test.pt')
    else:
        test_image_features = torch.empty((len(test_images), 768))
        batchify_run(process_image, test_images, test_image_features, 1024)

        torch.save(test_image_features,'img_embedding/test.pt')
    if os.path.exists('txt_embedding/'+'select_concept_'+shot+'shot'+str(num_concept)+'.pt'):
        text_features = torch.load('txt_embedding/'+'select_concept_'+shot+'shot'+str(num_concept)+'.pt')
    else:
        text_features = torch.empty((len(concepts),768))
        batchify_run(process_text, concepts, text_features, 1024)

    train_image_features /= train_image_features.norm(dim=-1, keepdim=True)
    test_image_features /= test_image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        attribute_activations_train = train_image_features @ text_features.t()
        attribute_activations_valid = test_image_features @ text_features.t()
    print(attribute_activations_train.shape)
    attribute_activations_train = attribute_activations_train.cpu().numpy()
    attribute_activations_valid= attribute_activations_valid.cpu().numpy()
    np.save('save_des/' + 'activation_train_'+shot+'shot'+str(num_concept)+'.npy', attribute_activations_train)
    np.save('save_des/' + 'activation_test_'+shot+'shot'+str(num_concept)+'.npy', attribute_activations_valid)


if __name__ == "__main__":
    import sys
    shot = sys.argv[1]
    num_concept = int(sys.argv[2])

    main(shot,num_concept)
