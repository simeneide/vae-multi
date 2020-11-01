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


#%% LOAD DATA
from pl_bolts.datamodules import CIFAR10DataModule
dataset = CIFAR10DataModule(data_dir=".")
batch =next(iter(dataset.train_dataloader()))
x, y = batch
imgsize = x.size()

num_classes = len(dataset.train_dataloader().dataset.dataset.classes)

#%% Load model
import models
vae = models.VAE(input_height=imgsize[2], num_classes=num_classes, use_label=True, lr=0.001)
#vae = vae.from_pretrained('cifar10-resnet18')

#%% Visualize reconstruction
import torchvision


#%%
from pytorch_lightning.callbacks import Callback
def inverse_normalize(
    tensor,
     mean = [x / 255.0 for x in [125.3, 123.0, 113.9]], 
     std = [x / 255.0 for x in [63.0, 62.1, 66.7]]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

class PlotImage(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the forward function for generation
    Requirements::
        # model must have img_dim arg
        model.img_dim = (1, 28, 28)
        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)

    """

    def __init__(self, batch= dict, num_samples: int = 32):
        super().__init__()
        self.batch = batch
        self.num_samples = num_samples
        
    def generate_image(self, trainer, pl_module):
        # Generate image
        dim = (self.num_samples, pl_module.hparams.latent_dim)
        z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)
        
        # generate images
        with torch.no_grad():
            pl_module.eval()
            images = pl_module.decoder(z)
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)

        images = inverse_normalize(images)
        grid = torchvision.utils.make_grid(images, normalize=True)
        str_title = f'{pl_module.__class__.__name__}_images_generated'
        
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

    def label2image(self, trainer, pl_module):
        # Extract z for label
        x, y = self.batch
        x = x.to(pl_module.device)
        y = y.to(pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            # Take mean vector:
            z = pl_module.classemb_mu.weight
            images = pl_module.decoder(z)
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)
        images = inverse_normalize(images)
        grid = torchvision.utils.make_grid(images, normalize=True)
        str_title = f'{pl_module.__class__.__name__}_images_label2img'
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

    def image2image(self, trainer, pl_module):
        # Extract z for label
        x, y = self.batch
        x = x.to(pl_module.device)
        y = y.to(pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            # Take mean vector:
            x = pl_module.encoder(x)
            z = pl_module.fc_mu(x)
            images = pl_module.decoder(z)
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)
        images = inverse_normalize(images)
        grid = torchvision.utils.make_grid(images, normalize=True)
        str_title = f'{pl_module.__class__.__name__}_images_img2img'
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)


    def reconstruct_images(self, trainer, pl_module):
        x, y = self.batch
        x = x.to(pl_module.device)
        y = y.to(pl_module.device)
        

        with torch.no_grad():
            pl_module.eval()
            out = pl_module(x, y)
            images = out['image']
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)
        images = inverse_normalize(images)
        grid = torchvision.utils.make_grid(images, normalize=True)
        str_title = f'{pl_module.__class__.__name__}_images_reconstructed'
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

    def on_epoch_end(self, trainer, pl_module):
        self.generate_image(trainer, pl_module)
        self.reconstruct_images(trainer, pl_module)
        self.label2image(trainer, pl_module)
        self.image2image(trainer, pl_module)
#%%
cb_imageplot = PlotImage(batch)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


cb_checkpoint = ModelCheckpoint(monitor='val/loss', verbose=True, save_last=True)
trainer = pl.Trainer(
    gpus=1, 
    #auto_lr_find=True,
    logger = TensorBoardLogger('lightning_logs', name='sgmcmc-lr0.1'),
    callbacks=[cb_imageplot],
    checkpoint_callback=cb_checkpoint)

#trainer.tune(vae, train_dataloader=dataset)
#%%
trainer.fit(vae, dataset)

# %% Analysis
vae = models.VAE.load_from_checkpoint("lightning_logs/version_14/checkpoints/epoch=348.ckpt")

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
