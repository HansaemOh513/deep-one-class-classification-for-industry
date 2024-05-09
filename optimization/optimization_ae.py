import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
sys.path.append('/home/canon/project/hansaem/python/occ')
from model.model_ae import Autoencoder
from torchsummary import summary
from sklearn.metrics import roc_auc_score
import logging

class AETrainer(Autoencoder):
    def __init__(self, device, weight_decay: float = 1e-6, lr = 0.001, batch_size = 32, max_iteration = 100, summary_option=False, lr_milestones = [30, 60, 90]):
        super().__init__()
        self.device = device
        self.model = Autoencoder()
        self.model = self.model.to(device)
        if summary_option:
            summary(self.model, (3, 256, 256)) # 수정해야 하지만 수정하기 귀찮아서 이렇게 내버려둠.
            print("optimization_ae.py, line 21")
            sys.exit()
        self.max_iteration = max_iteration
        self.lr_milestones = lr_milestones
        # self.data_size = data_size
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        
    def train(self, data):

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad = 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        start_time = time.time()
        self.model.train()
        
        for epoch in range(self.max_iteration):
            scheduler.step()
            if epoch in self.lr_milestones:
                print('LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            epoch_start_time = time.time()
            losses = []
            for i in range(0, data.shape[0], self.batch_size):
                inputs = data[i:i+self.batch_size, ...] # (batch, channel, width, height)

                # Zero the network parameter gradients
                optimizer.zero_grad()
                
                # Update network parameters via backpropagation: forward + backward + optimize
                outputs, latent = self.model(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
            losses = np.array(losses)
            ave_loss = np.mean(losses)
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print('Epoch : {} / {} Time : {:.3f} Loss : {:.5f}'.format(epoch + 1, self.max_iteration, epoch_train_time, ave_loss))

        pretrain_time = time.time() - start_time
        print('Pretraining time: %.3f' % pretrain_time)
        print('Finished pretraining.')

        return self.model

    def test(self, data):
        logger = logging.getLogger()
        logger.info('Testing autoencoder...')
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            inputs = data
            inputs = inputs.to(self.device)
            outputs, label = self.model(inputs)
            scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
            loss = torch.mean(scores)
            print("Test loss:",loss.item())
        test_time = time.time() - start_time
        print(test_time)
    def latent_center(self, data):
        latents = 0
        self.model.eval()
        for i in range(0, data.shape[0], self.batch_size):
            with torch.no_grad():
                inputs = data[i:i+self.batch_size, ...] # (batch, channel, width, height)
                outputs, latent = self.model(inputs)
                for j in range(latent.shape[0]):
                    latents = latents + latent[j]
        return latents / data.shape[0] # latents / data size
    def latent_statistical_measure(self, data, center):
        distances = torch.empty((0), device=self.device)
        self.model.eval()
        for i in range(0, data.shape[0], self.batch_size):
            with torch.no_grad():
                inputs = data[i:i+self.batch_size, ...] # (batch, channel, width, height)
                outputs, latent = self.model(inputs)
                for j in range(latent.shape[0]):
                    distance = torch.sqrt(torch.sum(center - latent[j])**2)
                    distance = distance.unsqueeze(0)
                    distances = torch.concatenate((distances, distance), axis=0)
        return distances

# trainer = AETrainer(device, 1000, summary_option=True)