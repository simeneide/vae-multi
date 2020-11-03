import torchvision
import torch
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
            images = pl_module.decoder(z)['image']
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)

        images = inverse_normalize(images)
        grid = torchvision.utils.make_grid(images, normalize=True)
        str_title = f'generated_img'
        
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
            z = pl_module.decoders_func['label'].linear.weight
            images = pl_module.decoder(z)['image']
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)
        images = inverse_normalize(images)
        grid = torchvision.utils.make_grid(images, normalize=True)
        str_title = f'label2img'
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

    def image2image(self, trainer, pl_module):
        # Extract z for label
        #x, y = self.batch
        #x = x.to(pl_module.device)
        #y = y.to(pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            # Take mean vector:
            
            out = pl_module.encoder({'image' : self.batch['image']})
            images = pl_module.decoder(out['z_sample'])['image']
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)
        images = inverse_normalize(images)
        grid = torchvision.utils.make_grid(images, normalize=True)
        str_title = f'img2img'
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

    def text2image(self, trainer, pl_module):
        # Extract z for label
        #x, y = self.batch
        #x = x.to(pl_module.device)
        #y = y.to(pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            # Take mean vector:
            
            out = pl_module.encoder({'text' : self.batch['text']})
            images = pl_module.decoder(out['z_sample'])['image']
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)
        images = inverse_normalize(images)
        grid = torchvision.utils.make_grid(images, normalize=True)
        str_title = f'text2img'
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

    def reconstruct_images(self, trainer, pl_module):
        #x, y = self.batch
        #batch = {'image' : x.to(pl_module.device), 'label' : y.to(pl_module.device)}
        

        with torch.no_grad():
            pl_module.eval()

            out = pl_module(self.batch)
            images = out['image']
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)
        images = inverse_normalize(images)
        grid = torchvision.utils.make_grid(images, normalize=True)
        str_title = f'reconstruct_img'
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

    def on_epoch_end(self, trainer, pl_module):
        try:
            self.batch = {key : val.to(pl_module.device) for key, val in self.batch.items()}
            self.generate_image(trainer, pl_module)
            self.reconstruct_images(trainer, pl_module)
            self.image2image(trainer, pl_module)
        except Exception as e:
            print(e)
        if self.batch.get("label") is not None:
            try:
                self.label2image(trainer, pl_module)
            except Exception as e:
                print(e)
        if self.batch.get("text") is not None:
            try:
                self.text2image(trainer, pl_module)
            except Exception as e:
                print(e)
