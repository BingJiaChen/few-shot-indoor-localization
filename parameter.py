import argparse

def parser():
    parser = argparse.ArgumentParser(description='Device-free indoor localization')
    parser.add_argument('--model',choices=['GNN','Attentive_GNN','EGNN','ChebyNet'],default='GNN',help='which model to use')
    parser.add_argument('--source_path',default='./CSI_data/EXP1.pickle',help='the data from source domain')
    parser.add_argument('--target_path',default='./CSI_data/EXP3.pickle',help='the data from target domain')
    parser.add_argument('--pretrain_model',default='./CSI_data/model_fine_tuning.ckpt',help='pretrain model to use')
    parser.add_argument('-seed',default=42069,type=int)
    parser.add_argument('--batch_size',default=16,type=int,help='size of data to use for each iteration')
    parser.add_argument('--lr',default=0.01,type=float,help='learning rate')
    parser.add_argument('--n_iter',default=6000,type=int,help='the total iterations in training')
    parser.add_argument('--log_interval',default=100,type=int,help='the period to print loss and accuracy in training')
    parser.add_argument('--log_eval',default=1000,type=int,help='the period to do validation')
    parser.add_argument('--beta',default=0.7,type=float,help='beta for Attentive GNN')
    parser.add_argument('--reg',default=0.01,type=float,help='lambda for Fuzzy GNN')
    parser.add_argument('--way',default=18,type=int,help='n-way of total locations')
    parser.add_argument('--shot',default=5,type=int,help='k-shot for each location')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--Kg',default=3,type=int,help='k-localized for ChebyNet')

    return parser.parse_args()

def print_args(args):
    for k, v in vars(args).items():
        print('{:<16}:{}'.format(k,v))
