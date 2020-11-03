#%%
# pip install pytorch-lightning-bolts ipywidgets test_tube --upgrade

# A script that should to variational autoencoders - after some time on a multi head.
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule
#%%

#%% LOAD DATA
import data
dataset = data.COCOCaptionDataModule()
#
##dataset = CIFAR10DataModule(data_dir=".")
batch =next(iter(dataset.train_dataloader()))
#x, y = batch
#%%
#num_classes = len(dataset.train_dataloader().dataset.dataset.classes)
#%% Load model
import models
vae = models.VAE(input_height=dataset.dims[2], num_labels=10, lr=0.001)

#%%
import utils
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
cb_imageplot = utils.PlotImage(batch)
cb_checkpoint = ModelCheckpoint(monitor='val/loss', verbose=True, save_last=True)
trainer = pl.Trainer(
    gpus=1, 
    #auto_lr_find=True,
    logger = TensorBoardLogger('lightning_logs', name='coco-withtext'),
    callbacks=[cb_imageplot],
    checkpoint_callback=cb_checkpoint)

#trainer.tune(vae, train_dataloader=dataset)
#%%
trainer.fit(vae, dataset)

# %% Analysis
#vae = models.VAE.load_from_checkpoint("lightning_logs/version_14/checkpoints/epoch=348.ckpt")

# %%
#%% Test batch

out = vae(x,y)
out['image'].min()
out = {key : val.detach() for key, val in out.items()}
#%%
plt.imshow(torchvision.utils.make_grid(x).permute(1,2,0))
plt.show()
g = torchvision.utils.make_grid(out['image'])#.permute(1,2,0)
plt.imshow(g)

trainer.logger.experiment.add_image("normalize", torchvision.utils.make_grid(out['image'], normalize=True), global_step=1)
trainer.logger.experiment.add_image("orig", torchvision.utils.make_grid(x), global_step=1)
plt.imshow(inverse_normalize(x)[0].permute(1,2,0).detach())

# %%
