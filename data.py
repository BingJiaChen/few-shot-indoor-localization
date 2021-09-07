from utils import data_preproc, count_data
import time
import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self,data,label=None):
        super(MyDataset,self).__init__()
        self.data = data
        self.label = label
    def __getitem__(self,idx):
        data = self.data[idx]
        if self.label is not None:
            label = self.label[idx]
            return data, label
        else:
            return data, 1
    def __len__(self):
        return len(self.data)
    

class MyDataLoader(Dataset):
  def __init__(self,source_path,target_path,train=True,nway=18,nshot=5):
    super(MyDataLoader,self).__init__()
    self.nway = nway
    self.nshot = nshot
    self.input_channel = 1
    self.size = 120
    self.full_data_dict, self.few_data_dict = self.load_data(source_path,target_path,train)

    print('full_data_num: %d' % count_data(self.full_data_dict))
    print('few_data_num: %d' % count_data(self.few_data_dict))
    
    
  def load_data(self,source_path,target_path,train):
    full_data_dict = {}
    few_data_dict = {}
    data_full, label_full = data_preproc(np.asarray(pickle.load(open(source_path,'rb'))))
    data_few, label_few = data_preproc(np.asarray(pickle.load(open(target_path,'rb'))))
    
    data_full = np.concatenate((data_full,data_few))
    temp_label = np.copy(label_few) + 16
    label_full = np.concatenate((label_full,temp_label))
    for i in range(len(label_full)):
        if label_full[i] not in full_data_dict:
            full_data_dict[label_full[i]] = [data_full[i]]
        else:
            full_data_dict[label_full[i]].append(data_full[i])

    for i in range(len(label_few)):
        if label_few[i] not in few_data_dict:
            few_data_dict[label_few[i]] = [data_few[i]]
        else:
            few_data_dict[label_few[i]].append(data_few[i])
    
    for i in range(16,34):
        full_data_dict[i] = full_data_dict[i][:self.nshot+1]
    return full_data_dict, few_data_dict

  def load_batch_data(self,train=True,batch_size=16,nway=18,num_shot=5):
    if train:
        data_dict = self.full_data_dict
    else:
        data_dict = self.few_data_dict
    
    x = []
    label_y = []
    one_hot_y = []
    class_y = []

    xi = []
    label_yi = []
    one_hot_yi = []

    map_label2class = []

    for i in range(batch_size):
        sampled_classes = random.sample(data_dict.keys(),nway)
        positive_class = random.randint(0,nway-1)
        label2class = torch.LongTensor(nway)

        single_xi = []
        single_one_hot_yi = []
        single_label_yi = []
        single_class_yi = []

        for j, n_class in enumerate(sampled_classes):
            if j == positive_class:
                sampled_data = random.sample(data_dict[n_class],num_shot+1)
                x.append(torch.from_numpy(sampled_data[0]))
                label_y.append(torch.LongTensor([j]))
                one_hot = torch.zeros(nway)
                one_hot[j] = 1.0
                one_hot_y.append(one_hot)
                class_y.append(torch.LongTensor([n_class]))
                shots_data = torch.Tensor(sampled_data[1:])
            else:
                shots_data = torch.Tensor(random.sample(data_dict[n_class],num_shot))
                
            single_xi += shots_data
            single_label_yi.append(torch.LongTensor([j]).repeat(num_shot))
            one_hot = torch.zeros(nway)
            one_hot[j] = 1.0
            single_one_hot_yi.append(one_hot.repeat(num_shot,1))

            label2class[j] = n_class
        
        shuffle_index = torch.randperm(num_shot*nway)
        xi.append(torch.stack(single_xi,0)[shuffle_index])
        label_yi.append(torch.cat(single_label_yi,dim=0)[shuffle_index])
        one_hot_yi.append(torch.cat(single_one_hot_yi,dim=0)[shuffle_index])
        map_label2class.append(label2class)

    return [torch.stack(x,0), torch.cat(label_y,dim=0), torch.stack(one_hot_y,0), torch.cat(class_y,dim=0), torch.stack(xi,0), torch.stack(label_yi,0), torch.stack(one_hot_yi,0), torch.stack(map_label2class,0)]
  
  def load_tr_batch(self,batch_size=16,nway=18,num_shot=5):
    return self.load_batch_data(True,batch_size,nway,num_shot)
  def load_te_batch(self,batch_size=16,nway=18,num_shot=5):
    return self.load_batch_data(False,batch_size,nway,num_shot)
  
  def get_data_list(self, data_dict):
    data_list = []
    label_list = []
    for i in data_dict.keys():
      for data in data_dict[i]:
        data_list.append(data)
        label_list.append(i)

    now_time = time.time()

    random.Random(now_time).shuffle(data_list)
    random.Random(now_time).shuffle(label_list)

    return data_list, label_list

  def get_full_data_dict(self):
    return self.get_data_list(self.full_data_dict)
  def get_few_data_dict(self):
    return self.get_data_list(self.few_data_dict)