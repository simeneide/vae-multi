from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

from pl_bolts.models.autoencoders.components import resnet18_encoder, resnet18_decoder
from pl_bolts.models.autoencoders.components import resnet50_encoder, resnet50_decoder


class VAE(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.

    Model is available pretrained on different datasets:

    Example::

        # not pretrained
        vae = VAE()

        # pretrained on cifar10
        vae = VAE.from_pretrained('cifar10-resnet18')

        # pretrained on stl10
        vae = VAE.from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18':
            'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/vae/vae-cifar10/checkpoints/epoch%3D89.ckpt',
        'stl10-resnet18':
            'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/vae/vae-stl10/checkpoints/epoch%3D89.ckpt'
    }

    def __init__(
        self,
        input_height: int,
        num_classes= int,
        use_label: bool = False,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height


        # Image ENC/DEC
        valid_encoders = {
            'resnet18': {'enc': resnet18_encoder, 'dec': resnet18_decoder},
            'resnet50': {'enc': resnet50_encoder, 'dec': resnet50_decoder},
        }

        self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1)
        self.decoder = valid_encoders[enc_type]['dec'](self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

        # Label ENC/DEC
        self.register_parameter("w",nn.Parameter(torch.tensor([0.5])))
        self.use_label = use_label
        self.num_classes = num_classes
        self.classemb_mu = nn.Embedding(self.num_classes, embedding_dim=self.latent_dim)
        self.classemb_var = nn.Embedding(self.num_classes, embedding_dim=self.latent_dim)
        self.decoder_class = nn.Linear(self.latent_dim, self.num_classes)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x, y=None):
        # Image
        x = self.encoder(x)
        mu_img = self.fc_mu(x)
        var_img = torch.exp(self.fc_var(x))
        
        # label
        if self.use_label:
            mu_label = self.classemb_mu(y)
            var_label = torch.exp(self.classemb_var(y))

            w1 = nn.Sigmoid()(self.w)
            w2 = 1.0 - w1

            mu = (w2*mu_label + w1*mu_img)
            std = ((w1**2 * var_img + w2**2 *var_label)).sqrt()
        else:
            mu = mu_img
            std = var_img.sqrt()
        p, q, z = self.sample(mu, std)

        out = {
            'label' : self.decoder_class(z),
            'image' : self.decoder(z)
            }

        return out

    def _run_step(self, x, y=None):
        # Image
        x = self.encoder(x)
        mu_img = self.fc_mu(x)
        var_img = torch.exp(self.fc_var(x))
        
        # label
        if self.use_label:
            mu_label = self.classemb_mu(y)
            var_label = torch.exp(self.classemb_var(y))

            w1 = nn.Sigmoid()(self.w)
            w2 = 1.0 -w1

            mu = (w2*mu_label + w1*mu_img)
            std = ((w1**2 * var_img + w2**2 * var_label)).sqrt()
            #mu = 0.5*(mu_label + mu_img)
            #std = (0.5*(var_img + var_label)).sqrt()
        else:
            mu = mu_img
            std = var_img.sqrt()
        p, q, z = self.sample(mu, std)
        out = {
            'p' : p,
            'q' : q,
            'z' : z,
            'label' : self.decoder_class(z),
            'image' : self.decoder(z)
            }
        return out

    def sample(self, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        out = self._run_step(x, y)

        image_loss = F.mse_loss(out['image'], x, reduction='mean')
        label_loss = F.cross_entropy(out['label'], y)

        log_qz = out['q'].log_prob(out['z'])
        log_pz = out['p'].log_prob(out['z'])

        kl = log_qz - log_pz
        kl = kl.mean()

        loss = kl + image_loss + label_loss

        logs = {
            "image_loss": image_loss,
            "label_loss" : label_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        self.log("param/weight", value=self.w)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val/{k}": v for k, v in logs.items()})

        return loss

    def configure_optimizers(self):
        #return torch.optim.SGD(self.parameters(), lr = self.lr)
        return OptimizerSGHMC(net=self,alpha = self.lr, nu=0.9, sgmcmc=False)
        #return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim", type=int, default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

    
def cli_main(args=None):
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule

    pl.seed_everything()

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "stl10", "imagenet"])
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule
    elif script_args.dataset == "stl10":
        dm_cls = STL10DataModule
    elif script_args.dataset == "imagenet":
        dm_cls = ImagenetDataModule
    else:
        raise ValueError(f"undefined dataset {script_args.dataset}")

    parser = VAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)
    args.input_height = dm.size()[-1]

    if args.max_steps == -1:
        args.max_steps = None

    model = VAE(**vars(args))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
    return dm, model, trainer


from torch.optim.optimizer import Optimizer

class OptimizerSGHMC(Optimizer):
    def __init__(self, net, alpha=1e-4, nu=1.0, sgmcmc=True):
        super(OptimizerSGHMC, self).__init__(net.parameters(), {})
        self.net = net
        self.sgmcmc = sgmcmc
        self.alpha = alpha
        self.nu = nu
        self.noise_std = (2*self.alpha*self.nu)**0.5

        self.momentum = {key : torch.zeros_like(par) for key, par in self.net.named_parameters()}
    
    @torch.no_grad()
    def step(self):
        for name, par in self.net.named_parameters():
            newpar = par + self.momentum[name]
            par.copy_(newpar)

            # Update momentum par:
            self.momentum[name] = (1-self.nu)*self.momentum[name] - self.alpha*par.grad
            if self.sgmcmc:
                noise = torch.normal(torch.zeros_like(self.momentum[name]), std=self.noise_std)
                self.momentum[name] += noise


if __name__ == "__main__":
    dm, model, trainer = cli_main()