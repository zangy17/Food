import numpy as np
import torch
from PIL import Image
import json
from tqdm import tqdm
import torch.nn as nn
import os
import open_clip
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset, DataLoader



model, train_preprocess, test_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai',cache_dir='../clip')

shot = sys.argv[1]
num_concept = int(sys.argv[2])


if shot=='1':
    batch_size = 16
    max_epoch = 100
    lr = 5e-4
elif shot == '2':
    batch_size = 32
    max_epoch = 100
    lr = 5e-4
elif shot == '4':
    batch_size = 64
    max_epoch = 100
    lr = 5e-4
elif shot =='8':
    batch_size = 128
    max_epoch = 100
    lr = 5e-4
else:
    batch_size = 256
    max_epoch = 100
    lr = 5e-4


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
model_text = TextCLIP(model)
model_image = ImageCLIP(model)

model_text = torch.nn.DataParallel(model_text)
model_image = torch.nn.DataParallel(model_image)
tokenizer = open_clip.get_tokenizer('ViT-L-14')



class CubDataset(Dataset):
    def __init__(self, images,labels,tokenizer):
        self.images=images
        self.labels=labels
        self.tokenize = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img, label = self.images[i], int(self.labels[i])
        return img,label,i


import json


train_labels = np.load('pred_labels_all.npy')
train_labels = torch.LongTensor(train_labels)
train_images = torch.load('data/'+'train_data_'+shot+'shot.pt')['image']
train_set = CubDataset(train_images,train_labels,tokenizer)

print(len(train_set))
#print(len(test_set))


device = 'cuda:0'

model = model.to(device)




train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,num_workers=32)

with open('all_att_90.json','r') as fp:
    texts = json.load(fp)
texts = tokenizer(texts)
texts = texts.to(device)



#test(ddp_model)
print("Training ImageNet...")

for (name,param) in model.named_parameters():
    if name!='text_projection' and name!='visual.proj' and name != 'logit_scale':
        param.requires_grad=False
    if name == 'text_projection':
        print(param)
        print(param.shape)
    if name == 'visual.proj':
        print(param)
        print(param.shape)
    if name == 'logit_scale':
        print(param)
params = filter(lambda p:p.requires_grad, model.parameters())
cls_truth = np.load('class_label_des_90.npy')
cls_truth = torch.Tensor(cls_truth).t()
cls_truth = cls_truth.to(device)
if not os.path.exists('checkpoint_'+shot+'shot'):
    os.mkdir('checkpoint_'+shot+'shot')
#fjirfgurhfh
optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
loss_func = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=max_epoch)
for epoch in range(max_epoch):

    model.train()
    total = 0.
    correct_cupl = 0.
    avg_loss = 0.

    for (images, targets, num) in tqdm(train_loader):

        images = images.to(device)
        targets = targets.to(device)


        optimizer.zero_grad()
        # predict
        # image_features = model_image(images)
        # image_features = image_features/image_features.norm(dim=-1, keepdim=True)

        # logits_base = image_features @ zeroshot_weights_base
        image_features = model_image(images)
        text_features = model_text(texts)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features,dim=-1)
        pred = torch.matmul(image_features, text_features.t()) * model.logit_scale
        pred = torch.matmul(pred,cls_truth)
        # logits_both = image_features @ zeroshot_weights_gpt_both
        loss = loss_func(pred,targets)
        loss.backward()
        loss_item = loss.detach().cpu().numpy()
        avg_loss += loss_item
        # convert_models_to_fp32(model)
        optimizer.step()
        total += len(images)

    #scheduler.step()
    print('epoch:', epoch, 'epoch loss:', np.mean(avg_loss / total * batch_size))
    if epoch %10==9:
        torch.save(model.state_dict(), 'checkpoint_'+shot+'shot/'+'model'+str(num_concept)+'_' + str(epoch) + '.pt')
