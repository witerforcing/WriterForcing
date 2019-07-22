import os
import argparse
import copy
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime
from sklearn.metrics import confusion_matrix
import matplotlib
from tqdm import tqdm


matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from io import open
import unicodedata
import string
import re
import random
import time
import math

def load_evalIters(args, model, data):
    optimizer = optim.SGD(model.parameters(), lr=args.learningrate, momentum=args.momentum)
    load_model_dict(args, model, optimizer, resume_train=True, model_type="best")
    #     optimizer = optim.Adam(model.parameters())

    pad_idx = data.TRG.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    valid_iterator = data.valid_iterator

    valid_loss = evaluate(epoch, model, valid_iterator, criterion, data)
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


def get_loss(output_probs, target_ids, criterion):
    loss = 0

    seq_len = target_ids.shape[1]
    distribution_len = len(output_probs)
    # assert(distribution_len == seq_len)

    for i in range(seq_len):
        loss += criterion(output_probs[i], target_ids[0, i].view(-1))

    return loss


def test_model(model, dataloader, mode='dev'):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, num_datapoints = 0.0, 0.0, 0.0
    results = []
    for i, sample_batched in enumerate(dataloader):
        batch, label = sample_batched[0].to(device), sample_batched[1].to(device)
        pred = model(batch)

        batch_loss = criterion(pred, label)
        loss += batch_loss.item()
        _, pred = pred.max(dim=1)
        acc += (pred == label).sum().float()
        num_datapoints += len(pred)
    #     print(num_datapoints)
    acc = acc / num_datapoints
    loss = loss / num_datapoints
    return loss, acc


def save_model(args, model, model_type="best"):
    SAVE_DIR = 'saved_models'
    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')

    torch.save(model.state_dict(), SAVE_DIR + f'/Story_{model_type}_{args.model_name}.pth')
    print('Finished saving model')


def eval_saved_model(args, model_name, type="best"):
    SAVE_DIR = 'saved_models'
    PATH = SAVE_DIR + f'/Story_{model_type}_{args.model_name}.pth'
    model = torch.load(PATH)
    model.eval()
    return model


def save_model_dict(args, model, optimizer, epoch, loss, model_type="best"):
    SAVE_DIR = 'saved_models'
    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')
    PATH = SAVE_DIR + f'/Story_{model_type}_{args.model_name}.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, PATH)


def load_model_dict(args, model, optimizer, resume_train=True, model_type="best"):
    SAVE_DIR = 'saved_models'
    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')
    PATH = SAVE_DIR + f'/Story_{model_type}_{args.model_name}.pt'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
    if resume_train:
        model.train()
    # todo add returns for optimizer epoch and loss if resuming

    return model

