# https://github.com/lukasruff/Deep-SVDD-PyTorch
import os
os.chdir('occ')
import sys
import argparse
import cv2
from optimization.optimization_ae import AETrainer
from dataloader import loader
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
os.chdir('../')
warnings.filterwarnings('ignore')
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--max_iteration', type=int, default=100, help='Iteration to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--dropout_rate', type=float, default=0.7, help='Dropout_rate [default: 0.7]')
parser.add_argument('--summary', type=str, default=False, help='Model summary [default: False]')
parser.add_argument('--schedule', type=str, default=True, help='Optimizer scheduler [default: True]')
parser.add_argument('--training_mode', type=str, default=False, help='if training mode is true, then it starts training and save parameters [default: False]')
parser.add_argument('--summary_option', type=str, default=False, help='model summary option [default: False]')
parser.add_argument('--example', type=str, default=False, help='See example figure [default: False]')

args = parser.parse_args()
device = torch.device('cuda:' + '{}'.format(args.gpu))
if args.training_mode:
    print("This is training mode")
else :
    print("This is not training mode")
# item = 'C550'

# MS_list =  [['3029C005AA', 'step_1'], ['3029C006AA', 'step_1'], ['3029C009AA', 'step_1'], ['3029C010AA', 'step_1'],
#             ['3030C002AA', 'step_1'], ['3030C003AA', 'step_1'], ['3030C004AA', 'step_1'], ['3031C001AA', 'step_1'],
#             ['3031C002AA', 'step_1'], ['3031C003AA', 'step_1']]

item = 'C551'
MS_list = [['3029C003AA', 'step_1'], ['3029C004AA', 'step_1'], ['3030C001AA', 'step_1']]

# item = '1525'
# MS_list = [['3029C003AA', 'step_5'], ['3029C004AA', 'step_5'], ['3029C005AA', 'step_5'], ['3029C006AA', 'step_5'], 
#            ['3030C001AA', 'step_5'],['3029C009AA', 'step_5'], ['3029C010AA', 'step_5'], ['3030C002AA', 'step_5'], 
#            ['3030C003AA', 'step_5'], ['3030C004AA', 'step_5'], ['3031C001AA', 'step_5'], ['3031C002AA', 'step_5'], 
#            ['3031C003AA', 'step_5'],
#            ['3029C003AA', 'step_7'], ['3029C004AA', 'step_7'], ['3029C005AA', 'step_7'], ['3029C006AA', 'step_7'], 
#            ['3030C001AA', 'step_7'], ['3029C009AA', 'step_7'], ['3029C010AA', 'step_7'], ['3030C002AA', 'step_7'], 
#            ['3030C003AA', 'step_7'], ['3030C004AA', 'step_7'], ['3031C001AA', 'step_7'], ['3031C002AA', 'step_7'], 
#            ['3031C003AA', 'step_7']]
master = cv2.imread("../../master/"+item+".jpg")
data_loader = loader(master) # (batch, height, width, channel)

data = loader.MS_loader(data_loader, MS_list=MS_list, resize=[True, 256, 256]) # Resize option, width, height
# (batch, height, width, channel)
data_shifted = np.moveaxis(data, -1, 1)
data_torch = torch.tensor(data_shifted / 255, dtype=torch.float32, requires_grad=True, device=device)
# pytorch : (batch, channel, height, width)
# cv2     : (batch, height, width, channel)
trainer = AETrainer(device, max_iteration = args.max_iteration, lr = args.learning_rate, summary_option = args.summary_option)
if args.training_mode:
    AEmodel = trainer.train(data_torch)
    model_name = item+'.pth'
    save = os.path.join('occ/neural_parameters/ae', model_name)
    # torch.save(AEmodel.state_dict(), save)
else:
    model_name = item+'.pth'
    load = os.path.join('occ/neural_parameters/ae', model_name)
    trainer.model.load_state_dict(torch.load(load))
    trainer.model.eval()
    AEmodel = trainer.model

if args.example:
    reconstruction = AEmodel(data_torch[0:1, ...])
    reconstruction_shifted = np.transpose(reconstruction[0, ...].cpu().detach().numpy(), (1, 2, 0))
    print(reconstruction_shifted.shape)
    plt.imshow(data[0, ...])
    plt.show('image1.jpg')
    plt.imshow(reconstruction_shifted)
    plt.show('image2.jpg')

# Latent space center and distance's mean and variance
# 1. Find Latent space center 
#    The latent space construct dimension : (128) It might be too much high dimension

center = trainer.latent_center(data_torch)
train_distances = trainer.latent_statistical_measure(data_torch, center)

train_mean = torch.mean(train_distances).item()
train_var  = torch.var(train_distances).item()
print("Training data mean : {:.2f} Training data variance : {:.2f}".format(train_mean, train_var))

item = 'C550'
MS_list = [['3029C005AA', 'step_1'], ['3029C006AA', 'step_1'], ['3029C009AA', 'step_1'], ['3029C010AA', 'step_1'], 
           ['3030C002AA', 'step_1'], ['3030C003AA', 'step_1'], ['3030C004AA', 'step_1'], ['3031C001AA', 'step_1'], 
           ['3031C002AA', 'step_1'], ['3031C003AA', 'step_1']]

valid_data = loader.MS_loader(data_loader, MS_list=MS_list, resize=[True, 256, 256]) # Resize option, width, height
valid_data_shifted = np.moveaxis(valid_data, -1, 1)
valid_data_torch = torch.tensor(valid_data_shifted / 255, dtype=torch.float32, requires_grad=True, device=device)

valid_distances = trainer.latent_statistical_measure(valid_data_torch, center)

valid_mean = torch.mean(valid_distances).item()
valid_var  = torch.var(valid_distances).item()
print("Test data mean : {:.2f} Test data variance : {:.2f}".format(valid_mean, valid_var))
