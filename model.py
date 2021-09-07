import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gen_Laplacian


class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
      
  def forward(self, x):
    return x

class pretrain(nn.Module):
  def __init__(self,in_dim=120,label_num=18,args=None):
    super(pretrain,self).__init__()
    self.args = args
    self.in_dim = in_dim
    self.ch = 32
    self.out = label_num
    self.C = nn.Sequential(
        nn.Conv1d(1,self.ch,5,2,2),
        nn.BatchNorm1d(self.ch),
        nn.ReLU(),
        nn.Conv1d(self.ch,self.ch,5,2,2),
        nn.BatchNorm1d(self.ch),
        nn.ReLU(),
        nn.Conv1d(self.ch,self.ch,5,2,2),
        nn.BatchNorm1d(self.ch),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(self.ch*15,32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32,self.out)
    )
  def forward(self,x):
    return self.C(x)

class Gconv(nn.Module):
  def __init__(self,in_dim,out_dim,use_bn=True,args=None):
    super(Gconv,self).__init__()
    self.weight = nn.Linear(in_dim,out_dim)
    self.args = args
    if use_bn:
      self.bn = nn.BatchNorm1d(out_dim)
    else:
      self.bn = None
  def forward(self,x,A):
    x_next = torch.matmul(A,x)
    x_next = self.weight(x_next)

    if self.bn is not None:
      x_next = torch.transpose(x_next,1,2)
      x_next = x_next.contiguous()
      x_next = self.bn(x_next)
      x_next = torch.transpose(x_next,1,2)
    return x_next

class Adj_layer(nn.Module):
  def __init__(self,in_dim,hidden_dim,ratio=[2,2,1,1],args=None):
    super(Adj_layer,self).__init__()
    self.args = args
    self.beta = self.args.beta
    module_list = []
    inner_list = []
    for i in range(len(ratio)):
      if i==0:
        module_list.append(nn.Conv2d(in_dim,hidden_dim*ratio[i],1,1))
      else:
        module_list.append(nn.Conv2d(hidden_dim*ratio[i-1],hidden_dim*ratio[i],1,1))
      module_list.append(nn.BatchNorm2d(hidden_dim*ratio[i]))
      module_list.append(nn.LeakyReLU())
    module_list.append(nn.Conv2d(hidden_dim*ratio[-1],1,1,1))

    self.module_list = nn.ModuleList(module_list)

  def forward(self,x):
    V = x.shape[1]
    x_i = x.unsqueeze(2)
    x_j = torch.transpose(x_i,1,2)

    phi = torch.abs(x_i-x_j)
    phi = torch.transpose(phi,1,3)

    A = phi
    for l in self.module_list:
      A = l(A)
    
    A = torch.transpose(A,1,3)
    A = F.softmax(A,2).squeeze(3)
    
    if self.args.model=='Attentive_GNN':
      index = torch.topk(A,int(self.beta*V),dim=2)[1]
      mask_A = torch.zeros(A.shape).to(self.args.device)
      
      for i, ind in enumerate(index):

        col = torch.arange(self.args.way*self.args.shot+1).reshape(-1,1).expand(self.args.way*self.args.shot+1,int(self.beta*V)).reshape(-1)
        ind = ind.reshape(-1)
        mask_A[i][col,ind] = A[i][col,ind]
      
      return mask_A

    return A

class GNN_module(nn.Module):
  def __init__(self,nway,in_dim,hidden_dim,num_layers,feature='dense',args=None):
    super(GNN_module,self).__init__()
    self.args = args
    self.feature_type = feature
    adj_list = []
    GNN_list = []

    ratio = [2,1]

    if self.feature_type == 'dense':
      for i in range(num_layers):
        adj_list.append(Adj_layer(in_dim+hidden_dim//2*i,hidden_dim,ratio,self.args))
        GNN_list.append(Gconv(in_dim+hidden_dim//2*i,hidden_dim//2,args=self.args))

      last_adj = Adj_layer(in_dim+hidden_dim//2*num_layers,hidden_dim,ratio,self.args)
      last_conv = Gconv(in_dim+hidden_dim//2*num_layers,nway,use_bn=False,args=self.args)

    self.adj_list = nn.ModuleList(adj_list)
    self.GNN_list = nn.ModuleList(GNN_list)
    self.last_adj = last_adj
    self.last_conv = last_conv

  def forward(self,x):
    for i, _ in enumerate(self.adj_list):
      adj_layer = self.adj_list[i]
      conv_block = self.GNN_list[i]

      A = adj_layer(x)
      x_next = conv_block(x,A)
      x_next = F.leaky_relu(x_next,0.1)

      x = torch.cat([x,x_next],dim=2)

    A = self.last_adj(x)
    out = self.last_conv(x,A)

    return out[:,0,:], A

class GNN(nn.Module):
  def __init__(self,cnn_feature_size,gnn_feature_size,nway,args=None):
    super(GNN,self).__init__()
    self.args = args
    num_inputs = cnn_feature_size + nway
    gnn_layer = 2
    self.gnn = GNN_module(nway,num_inputs,gnn_feature_size,gnn_layer,'dense',self.args)

  def forward(self,inputs):
    logits, A = self.gnn(inputs)
    logits = logits.squeeze(-1)

    return logits, A

class gnnModel(nn.Module):
  def __init__(self,nway,args=None):
    super(gnnModel,self).__init__()
    self.args = args
    cnn_feature_size = 32
    gnn_feature_size = 16
    self.cnn_feature = pretrain(self.args)
    self.gnn = GNN(cnn_feature_size,gnn_feature_size,nway,self.args)
    self.fusion = nn.Conv2d(2,1,1,1)

  def forward(self,data):
    [x,_,_,_,xi,_,one_hot_yi,_] = data
    
    z = self.cnn_feature(x)
    zi_s = [self.cnn_feature(xi[:,i,:,:]) for i in range(xi.size(1))]
    zi_s = torch.stack(zi_s,dim=1)
    
    uniform_pad = torch.FloatTensor(one_hot_yi.size(0),1,one_hot_yi.size(2)).fill_(1.0/one_hot_yi.size(2)).to(self.args.device)
    labels = torch.cat([uniform_pad,one_hot_yi],dim=1)
    features = torch.cat([z.unsqueeze(1),zi_s],dim=1)
    
    nodes_features = torch.cat([features,labels],dim=2)

    out_logits, A = self.gnn(nodes_features)
    logits_prob = F.log_softmax(out_logits,dim=1)

    return logits_prob, A


class ChebConv(nn.Module):
  def __init__(self,Kg,in_features,out_features,enble_bias=True,args=None):
    super(ChebConv,self).__init__()
    self.args = args
    self.Kg = Kg
    self.in_features = in_features
    self.out_features = out_features
    self.act = nn.ReLU()
    self.L = nn.Linear(Kg*in_features,out_features,bias=enble_bias)

  def forward(self,x,lap):
    ns = lap.shape[1]
    bs = lap.shape[0]

    x_0 = x
    x_1 = torch.bmm(lap,x)
    if self.Kg == 1:
      x_list = [x_0]

    elif self.Kg == 2:
      x_list = [x_0,x_1]

    elif self.Kg >= 3:
      x_list = [x_0,x_1]
      for k in range(2,self.Kg):
        x_list.append(torch.bmm(2*lap,x_list[k-1])-x_list[k-2])
    
    x_tensor = torch.stack(x_list,dim=-1).reshape((bs,ns,-1))
    
    x_out = self.L(x_tensor)
    
    return x_out

class ChebNet(nn.Module):
  def __init__(self,Kg,n_feat,n_hid,n_class,args=None):
    super(ChebNet,self).__init__()
    self.args = args
    self.n_hid = n_hid
    self.n_class = n_class
    self.Kg = Kg
    adj_list = []
    conv_list = []
    ratio = [2,1]
    for i in range(len(ratio)):
      adj_list.append(Adj_layer(n_feat+n_hid//2*i,n_hid,ratio,self.args))
      conv_list.append(ChebConv(self.Kg,n_feat+n_hid//2*i,n_hid//2,args=self.args))

    last_adj = Adj_layer(n_feat+n_hid//2*len(ratio),n_hid,ratio,self.args)
    last_conv = ChebConv(self.Kg,n_feat+n_hid//2*len(ratio),n_class,False,args=self.args)

    self.adj_list = nn.ModuleList(adj_list)
    self.conv_list = nn.ModuleList(conv_list)
    self.last_adj = last_adj
    self.last_conv = last_conv

    self.cnn_feature = pretrain()
    self.act = nn.LeakyReLU()
    self.log_softmax = nn.LogSoftmax(dim=1)

  def forward(self,data):
    [x,_,_,_,xi,_,one_hot_yi,_] = data

    z = self.cnn_feature(x)
    zi_s = [self.cnn_feature(xi[:,i,:,:]) for i in range(xi.size(1))]
    zi_s = torch.stack(zi_s,dim=1)
    uniform_pad = torch.FloatTensor(one_hot_yi.size(0),1,one_hot_yi.size(2)).fill_(1.0/one_hot_yi.size(2)).to(self.args.device)
    labels = torch.cat([uniform_pad,one_hot_yi],dim=1)
    features = torch.cat([z.unsqueeze(1),zi_s],dim=1)
    x = torch.cat([features,labels],dim=2)

    for i in range(len(self.adj_list)):
      adj_layer = self.adj_list[i]
      conv_layer = self.conv_list[i]

      A = adj_layer(x)
      lap = gen_Laplacian(A.clone())
      x_next = conv_layer(x,lap)
      x_next = self.act(x_next)
      x = torch.cat([x,x_next],dim=2)

    A = self.last_adj(x)
    lap = gen_Laplacian(A.clone())
    x = self.last_conv(x,lap)
    out = x[:,0,:]
    
    logits_prob = self.log_softmax(out)


    return logits_prob, A