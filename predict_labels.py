import sys

import numpy as np
import torch
import clip

from PIL import Image

import json
from tqdm import tqdm
import torch.nn as nn
import open_clip
data_root = 'data/'
shots = ['all']
num_concept = int(sys.argv[1])
device = 'cuda:0'
class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)
model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-L-14')
model_text = TextCLIP(model)
model_text.eval()
model_text.to(device)

for shot in shots:
    train_imgs = torch.load('img_embedding/train_'+shot+'shot.pt')
    train_y = torch.load(data_root + 'train_data_' + shot + 'shot.pt')['label']
    train_imgs = train_imgs.to(device)
    train_y = train_y.to(device)
    with open('full_concept_90.json', 'r') as fp:
        prompts = json.load(fp)
    textnames = list(prompts.keys())

    with torch.no_grad():
        zeroshot_weights = []
        i = 0
        for classname in tqdm(textnames):

            texts = []

            for t in prompts[textnames[i]]:
                texts.append(t)
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            i += 1

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        zeroshot_weights = torch.squeeze(zeroshot_weights)
        print(train_imgs.shape)
        print(zeroshot_weights.shape)
        predicts = train_imgs@zeroshot_weights
        print(predicts.shape)
        predict_labels = torch.argmax(predicts, dim=1)
        correct = torch.sum(predict_labels==train_y)
        print(correct/train_y.shape[0])
        pred_labels = predict_labels.cpu().numpy()
        np.save('pred_labels_all.npy',pred_labels)


