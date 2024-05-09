# The original paper of this repository is Ruff, Lukas, et al. "Deep one-class classification." International conference on machine learning. PMLR, 2018.
https://proceedings.mlr.press/v80/ruff18a.html
https://github.com/lukasruff/Deep-SVDD-PyTorch

### Environment
torch : 2.2.2+cu118
torchaudio : 2.2.2+cu118
torchvision : 0.17.2+cu118
numpy : 1.24.1
opencv-python : 4.9.0.80
opencv-contrib-python : 4.9.0.80
matplotlib : 3.8.4
torchsummary : 1.4

### Train and test
To train SVDD, type next prompt.
python train_SVDD.py --training_mode 1
To test SVDD, type next prompt.
python train_SVDD.py --training_mode 0

### ordering
This file make paths clear.

### dataloader
This file plays the role of loading data.

### optimization
This folder contains two files, each tasked with training an autoencoder and SVDD, respectively.

### model
This folder comprises two files, each designated for modeling an autoencoder and SVDD, respectively.
