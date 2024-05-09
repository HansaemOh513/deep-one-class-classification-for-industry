# https://github.com/lukasruff/Deep-SVDD-PyTorch
import os
import sys
import argparse
from optimization.optimization_SVDD import DeepSVDDTrainer
from dataloader import loader
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--max_iteration', type=int, default=100, help='Iteration to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
parser.add_argument('--dropout_rate', type=float, default=0.7, help='Dropout_rate [default: 0.7]')
parser.add_argument('--summary', type=str, default=False, help='Model summary [default: False]')
parser.add_argument('--schedule', type=str, default=True, help='Optimizer scheduler [default: True]')
parser.add_argument('--training_mode', type=int, default=0, help='if training mode is 1, then it starts training and save parameters [default: 0]')
parser.add_argument('--summary_option', type=str, default=False, help='model summary option [default: False]')
args = parser.parse_args()
device = torch.device('cuda:' + '{}'.format(args.gpu))
if args.training_mode==1:
    print("This is training mode")
else :
    print("This is test mode")
def SVDD_training(item, MS_list):
    data_loader = loader(master)
    data = data_loader.MS_loader(MS_list, resize=[True, 256, 256]) # (batch, height, width, channel) RGB
    data_size = data.shape[0]
    data = np.moveaxis(data, -1, 1)
    data = torch.tensor(data / 255, dtype=torch.float32, requires_grad=True, device=device) # 여기서 정규화를 진행
    # pytorch : (batch, channel, height, width) RGB
    # cv2     : (batch, height, width, channel) BGR
    trainer = DeepSVDDTrainer(device, data_size, max_iteration = args.max_iteration, lr = args.learning_rate, summary_option=args.summary_option)
    SVDDmodel = trainer.train(data)
    model_name = item+'.pth'
    save = os.path.join('occ/neural_parameters/SVDD', model_name)
    save_c = os.path.join('occ/neural_parameters/SVDD/center', model_name)
    torch.save(SVDDmodel.state_dict(), save)
    torch.save(trainer.c, save_c)

def SVDD_test(item_normal, MS_list_normal, item_anomaly, MS_list_anomaly):
    data_loader = loader(master)
    data_normal = data_loader.MS_loader(MS_list_normal, resize=[True, 256, 256]) # (batch, height, width, channel) RGB
    data_normal_size = data_normal.shape[0]
    data_normal = np.moveaxis(data_normal, -1, 1)
    data_normal = torch.tensor(data_normal / 255, dtype=torch.float32, requires_grad=True, device=device)
    data_anomaly = data_loader.MS_loader(MS_list_anomaly, resize=[True, 256, 256]) # (batch, height, width, channel) RGB
    data_anomaly_size = data_anomaly.shape[0]
    data_anomaly = np.moveaxis(data_anomaly, -1, 1)
    data_anomaly = torch.tensor(data_anomaly / 255, dtype=torch.float32, requires_grad=True, device=device)
    # pytorch : (batch, channel, height, width)
    # cv2     : (batch, height, width, channel)
    trainer = DeepSVDDTrainer(device, 0, max_iteration = args.max_iteration, lr = args.learning_rate, summary_option=args.summary_option)
    model_name = item_normal+'.pth'
    load = os.path.join('occ/neural_parameters/SVDD', model_name)
    load_c = os.path.join('occ/neural_parameters/SVDD/center', model_name)
    trainer.model.load_state_dict(torch.load(load))
    trainer.c = torch.load(load_c).to(device)
    trainer.model.eval()
    
    # test mode
        
    dists = []
    for i in range(0, data_normal_size, trainer.batch_size):
        inputs = data_normal[i:i+trainer.batch_size, ...]
        outputs = trainer.model(inputs)
        expanded_c = trainer.c.repeat(outputs.shape[0], 1)
        dist = torch.sum((outputs - expanded_c) ** 2, dim=1)
        dists.append(np.mean(dist.cpu().detach().numpy()))
    dists = np.array(dists)
    print("Trained image dist : {:.5f}".format(np.mean(dists)))
    pass_dists = dists
    dists = []
    for i in range(0, data_anomaly_size, trainer.batch_size):
        inputs = data_anomaly[i:i+trainer.batch_size, ...]
        outputs = trainer.model(inputs)
        expanded_c = trainer.c.repeat(outputs.shape[0], 1)
        dist = torch.sum((outputs - expanded_c) ** 2, dim=1)
        dists.append(np.mean(dist.cpu().detach().numpy()))
    dists = np.array(dists)
    fail_dists = dists
    print("Test image dist : {:.5f}".format(np.mean(dists)))
    pass_mean  = np.mean(pass_dists)
    pass_var   = np.var(pass_dists)

    fail_mean  = np.mean(fail_dists)
    fail_var   = np.var(fail_dists)

    print("pass mean : {:.8f} pass var : {:.8f} fail mean : {:.8f} fail var : {:.8f}".format(pass_mean, pass_var, fail_mean, fail_var))

    plt.figure(figsize=(10, 6))
    plt.hist(pass_dists, bins=50, alpha=0.7, label='Pass Distances')
    plt.hist(fail_dists, bins=50, alpha=0.7, label='Fail Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of distances')
    plt.legend()
    plt.grid(True)
    plt.show()
item_normal = 'C550' # NPG
MS_list_normal = [['3029C005AA', 'step_1'], ['3029C006AA', 'step_1'], ['3029C009AA', 'step_1'], ['3029C010AA', 'step_1'], 
                  ['3030C002AA', 'step_1'], ['3030C003AA', 'step_1'], ['3030C004AA', 'step_1'], ['3031C001AA', 'step_1'], 
                  ['3031C002AA', 'step_1'], ['3031C003AA', 'step_1']]

item_anomaly = 'C551' # C-EXV
MS_list_anomaly = [['3029C003AA', 'step_1'], ['3029C004AA', 'step_1'], ['3030C001AA', 'step_1']]

master = cv2.imread("../master/C550/master.jpg")
if args.training_mode==1:
    SVDD_training(item_normal, MS_list_normal)
elif args.training_mode==0:
    SVDD_test(item_normal, MS_list_normal, item_anomaly, MS_list_anomaly)