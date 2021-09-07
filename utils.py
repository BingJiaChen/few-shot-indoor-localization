import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import torch

def single_minmaxscale(data, scale_range):
    def minmaxscale(data, scale_range):
        scaler = MinMaxScaler(scale_range)
        scaler.fit(data)
        normalized = scaler.transform(data)
        return normalized

    X = []
    for i in data:
        X.append(minmaxscale(i.reshape(-1,1), scale_range))
    return np.asarray(X)     


def data_preproc(dataset, scale_range = (-1, 1)):
    X_tra, y_tra, X_tst, y_tst = dataset
    X_tra = single_minmaxscale(X_tra, scale_range)
    X_tst = single_minmaxscale(X_tst, scale_range)

    X_tra = X_tra.astype('float32')
    X_tra = X_tra.reshape(-1,1,120)
    X_tst = X_tst.astype('float32')
    X_tst = X_tst.reshape(-1,1,120)
    data = np.concatenate((X_tra,X_tst))
    label = np.concatenate((y_tra,y_tst))
    print('Finished preprocessing.')
    
    return data, label

def count_data(data_dict):
    num = 0
    for key in data_dict.keys():
        num += len(data_dict[key])
    return num


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def count_acc(logits, label):
    pred = torch.argmax(F.softmax(logits), dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def label2edge(label):
    ns = label.shape[1]
    label = label.unsqueeze(-1).repeat(1, 1, ns)
    edge = torch.eq(label, label.transpose(1,2)).float()
    return edge

def gen_Laplacian(adj):
    ns = adj.shape[1]
    bs = adj.shape[0]
    iden = torch.eye(ns)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iden = iden.unsqueeze(0).repeat(bs,1,1).to(device)
    adj[iden.bool()] = 0
    D = torch.diag_embed(torch.pow(adj.sum(axis=2),-0.5))
    sym_norm_lap = iden - torch.bmm(D,torch.bmm(adj,D))
    e = torch.linalg.eigvals(sym_norm_lap)
    e_max = torch.max(torch.abs(e),1).values
    wid_sym_norm_lap = torch.mul((2/e_max).view(bs,1,1),sym_norm_lap) - iden
    return wid_sym_norm_lap