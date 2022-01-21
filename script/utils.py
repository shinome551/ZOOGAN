#!/usr/bin/env python
# coding: utf-8

import time

import numpy as np
import numpy.linalg as LA
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def getClassBalancedIndex(idx_list, targets):
    uniq, count = np.unique(targets[idx_list], return_counts=True)
    num_classes = len(uniq)
    max_cnt = count.max()
    sample_idx = -1 * np.ones((max_cnt, num_classes), dtype=np.int64)
    for c in range(num_classes):
        sample_idx[:count[c], c] = idx_list[targets[idx_list] == c]
    sample_idx = sample_idx.flatten()
    sample_idx = sample_idx[sample_idx >= 0]
    return sample_idx


def getModelResponse(pipe, embedset, device, normalize=False):
    loader = DataLoader(embedset, batch_size=100, shuffle=False)
    pipe.eval()
    key = ['index', 'label', 'prediction', 'confidence', 'loss']
    res = []
    embed = []
    def getEmbedding():
        def hook(model, input, output):
            embed.append(input[0])
        return hook
    handle = getattr(pipe.model, 'fc').register_forward_hook(getEmbedding())

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = pipe(inputs)
            conf, pred = torch.topk(outputs.softmax(-1), k=1, dim=1)
            loss = F.cross_entropy(outputs, labels, reduction='none')
            
            values = torch.hstack((labels.reshape(-1,1), pred.reshape(-1,1), conf.reshape(-1,1), loss.reshape(-1,1), ))
            values = values.cpu().tolist()
            for i in range(len(values)):
                res.append(dict(zip(key,[len(res)] + values[i])))

    embed = torch.vstack(embed).cpu().numpy()
    if normalize:
        embed = embed / np.linalg.norm(embed, axis=1, keepdims=True)
        
    handle.remove()

    return res, embed


def getLossDescendingOrder(res, pre_sample_idx, sample_num, incremental=False):
    res = sorted(res, key=lambda x: x['loss'])[::-1]
    if incremental:
        new_idx = [d_iter['index'] for d_iter in res if d_iter['index'] not in pre_sample_idx]
        sample_idx = np.concatenate([pre_sample_idx, new_idx[:sample_num]], axis=0)
    else:
        sample_idx = [d_iter['index'] for d_iter in res if d_iter['index'] in pre_sample_idx][:sample_num]
    return sample_idx


def repeatSampleidx(sample_idx, target_num):
    ite = target_num // len(sample_idx)
    sample_idx_rep = np.tile(sample_idx, ite + 1)[:target_num]
    return sample_idx_rep


class DatasetwithValues(Dataset):
    def __init__(self, dataset, values):
        self.dataset = dataset
        self.values = values
        self.data = dataset.data
        self.classes = dataset.classes
        
    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        value = self.values[idx]
        return data, target, value

    def __len__(self):
        return len(self.dataset)


def pruneData(embed, pre_sample_idx):
    start = time.time()
    embed_center = embed[pre_sample_idx] - embed[pre_sample_idx].mean(axis=0, keepdims=True)

    def update(invA, b):
        invA_new = invA - (invA @ b) @ (b.T @ invA) / (b.T @ invA @ b - 1)
        return invA_new

    iter_num = len(pre_sample_idx)
    invCorrMat = LA.inv(embed_center.T @ embed_center)
    sample_idx = []
    record = []
    abs_idx = pre_sample_idx
    for i in range(iter_num):
        X_bar = (invCorrMat @ embed_center.T).T
        R = X_bar / np.power(LA.norm(X_bar, axis=1, keepdims=True), 2)
        r_norm = LA.norm(R, axis=1)
        del_idx = r_norm.argmin()
        
        sample_idx.append(abs_idx[del_idx])
        invCorrMat = update(invCorrMat, embed_center[del_idx][:, None])

        d = {
            'iter': i,
            'index': abs_idx[del_idx],
            'value': round(r_norm[del_idx], 3),
            'invdet': LA.det(invCorrMat),
            'invtrace': np.trace(invCorrMat)
        }
        print(d, end='\r')
        record.append(d)

        embed_center = np.delete(embed_center, del_idx, 0) 
        abs_idx = np.delete(abs_idx, del_idx, 0) 

    sample_idx = np.array(sample_idx[::-1])

    print('')
    print(f'pruning finished / time:{time.time() - start:.3f}s')

    return sample_idx, record