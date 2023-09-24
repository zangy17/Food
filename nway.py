import os
import numpy as np
import pandas as pd
import torch
import open_clip
import argparse
import pdb

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from torch.utils.data import DataLoader
from PIL import Image



def main(shot,num_concept):

    data_root = 'data/'
    train_y = torch.load(data_root + 'train_data_' + shot + 'shot.pt')['label']
    test_y = torch.load(data_root + 'test_data.pt')['label']
    tr_att_activations = torch.Tensor(np.load('save_des/' + 'activation_train_'+shot+'shot'+str(num_concept)+'.npy'))
    t_att_activations = torch.Tensor(np.load('save_des/' + 'activation_test_'+shot+'shot'+str(num_concept)+'.npy'))
    tr_att_activations /= tr_att_activations.mean(dim=-1,keepdim=True)
    t_att_activations /= t_att_activations.mean(dim=-1,keepdim=True)

    cls_truth = np.load('class_label_des_'+shot+'shot'+str(num_concept)+'.npy')
    tr_ground_truth = cls_truth[train_y]
    t_ground_truth = cls_truth[test_y]

    print(t_att_activations.shape)
    print(len(t_ground_truth))


    rr = []

    # ConceptCLIP - Primitive (Logistic Regression)
    classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
    classifier.fit(tr_att_activations.numpy(), train_y.numpy())
    lr_score = classifier.score(t_att_activations.numpy(), test_y.numpy())
    rr.append(lr_score)
    # Full - Intervene (Logistic Regression)
    lr_score = classifier.score(t_ground_truth, test_y.numpy())
    rr.append(lr_score)
    print(rr)

if __name__ == "__main__":
    import sys

    shot = sys.argv[1]
    num_concept = int(sys.argv[2])
    main(shot,num_concept)

