import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
sys.path.append('/home/canon/project/hansaem/python/occ')
from model.model_SVDD import SVDD
from torchsummary import summary
from sklearn.metrics import roc_auc_score
import logging

class DeepSVDDTrainer(SVDD):
    def __init__(self, device, data_size, weight_decay: float = 1e-6, lr = 0.001, batch_size = 32, max_iteration = 100, summary_option=False, lr_milestones = [50]):
        super().__init__()
        self.device = device
        self.model = SVDD()
        self.model = self.model.to(device)
        if summary_option:
            summary(self.model, (3, 256, 256)) # 수정해야 하지만 수정하기 귀찮아서 이렇게 내버려둠.
            print("optimization_SVDD.py, line 21")
            sys.exit()
        self.objective ='one-class'
        self.max_iteration = max_iteration
        self.lr_milestones = lr_milestones
        self.data_size = data_size
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        # Deep SVDD parameters
        R = 0.0
        c = None
        nu = 0.1 # 하이퍼 파라미터.
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, data):
        logger = logging.getLogger()
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad='amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(data, self.model)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        self.model.train()
        for epoch in range(self.max_iteration):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_start_time = time.time()
            losses = []
            for i in range(0, self.data_size, self.batch_size):
                inputs = data[i:i+self.batch_size, ...]

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = self.model(inputs)
                # self.expanded_c = self.c.repeat(self.batch_size, 1) # batch 사이즈가 안나누어 떨어져서 맞아서 코드를 바꿈
                self.expanded_c = self.c.repeat(outputs.shape[0], 1)
                dist = torch.sum((outputs - self.expanded_c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                losses.append(loss.item())
            losses = np.array(losses)
            ave_loss = np.mean(losses)
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print('Epoch : {} / {} Time : {:.3f} Loss : {:.5f}'.format(epoch + 1, self.max_iteration, epoch_train_time, ave_loss))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return self.model

    def init_center_c(self, data, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for i in range(0, self.data_size, self.batch_size):
                inputs = data[i:i+self.batch_size, ...]
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
