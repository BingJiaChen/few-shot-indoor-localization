from trainer import Trainer

import os
import time
import random
import numpy as np
import torch
from parameter import parser, print_args

def main(args):
    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(args)
    trainer.load_pretrain(args.pretrain_model)
    trainer.train()

if __name__ == '__main__':
    args = parser()
    # print_args(args)
    main(args)