from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _train(self, x, y, config):
        # 학습 모드를 정해줘야 함 train/eval
        self.model.train()   

        # Shuffle and split before begin.
        # index를 shuffling함
        indices = torch.randperm(x.size(0), device=x.device)  

        # index 대로 다시 재배열함 and batch_size로 split해줌 
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)   
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)   

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)   

            # y_i.shape가 (batch_size,1)로 되어있다면 -> (bs,)로 바꿔주기 위해 sqeeze() -> loss function에 넣어줌
            loss_i = self.crit(y_hat_i, y_i.squeeze())    

            # Initialize the gradients of the model.
            # weight parameter의 gradient를 0으로 만들어줌
            self.optimizer.zero_grad()   
            
            # backpropagation 진행
            loss_i.backward()   

            # 외부에서 받아온 optimizer의 step 진행
            self.optimizer.step()   

            #--- weight parameter가 1번 update됨 ---#

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            # float를 안씌우면 tensor로 되서 loss에 연산이 다 물려버림(엄청난 메모리 leak 발생)
            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        # 모드 설정
        self.model.eval()  

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad():
            # Shuffle before begin.
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            # 데이터를 받아와서 넣어줌-> [(batch_size, 784), (batch_size,)]
            # 학습을 해서 각 epoch에 평균 loss를 구해줌
            train_loss = self._train(train_data[0], train_data[1], config)   
            valid_loss = self._validate(valid_data[0], valid_data[1], config)   

            # You must use deep copy to take a snapshot of current best weights.
            # 이전 loss와 비교해서 갱신을 시킬지 말지 결정해서 best model을 들고 있음
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore to best model.
        # best 모델을 restore함
        self.model.load_state_dict(best_model)
