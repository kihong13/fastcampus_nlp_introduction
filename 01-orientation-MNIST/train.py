import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer
from utils import load_mnist

def define_argparser():
    # bash/cmd 등에서 py를 실행시킬때 여러가지 인자를 줄 수 있음
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', required=True)# weight file의 이름
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1) # GPU ID
    p.add_argument('--train_ratio', type=float, default=.8) # train/validset ratio
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    config = p.parse_args() # config.gpu_id 이런식으로 실행시킴

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    # Data loader
    x, y = load_mnist(is_train=True)

    # Reshape tensor to chunk of 1-d vectors. : (101,28,28) -> (101,784)
    x = x.view(x.size(0), -1)

    # train/test size 결정
    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0) # suffling -> GPU -> split
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0) 

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    model = ImageClassifier(28**2, 10).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    trainer = Trainer(model, optimizer, crit)

    trainer.train((x[0], y[0]), (x[1], y[1]), config)

    # Save best model weights.
    torch.save({
        'model': trainer.model.state_dict(),   # key value dictionary로 만들어서 저장함
        'config': config,
    }, config.model_fn)

if __name__ == '__main__':   # 실행시킬때 구동되는 부분
    config = define_argparser()   # 받아온 입력값들을 정리함
    main(config)  # 정리한 값들(config)를 main에 넣어서 실행시킴
