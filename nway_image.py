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



def main(shot):

    data_root = 'data/'
    train_y = torch.load(data_root + 'train_data_' + shot + 'shot.pt')['label']
    test_y = torch.load(data_root + 'test_data.pt')['label']
    tr_att_activations = torch.load('img_embedding/train_'+shot+'shot.pt')
    t_att_activations = torch.load('img_embedding/test.pt')


    print(t_att_activations.shape)

    # ConceptCLIP - Primitive (Logistic Regression)
    classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
    classifier.fit(tr_att_activations.numpy(), train_y.numpy())
    lr_score = classifier.score(t_att_activations.numpy(), test_y.numpy())

    print(lr_score)

if __name__ == "__main__":
    import sys

    shot = sys.argv[1]
    main(shot)

