from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

from pl_bolts.models.autoencoders.components import resnet18_encoder, resnet18_decoder
from pl_bolts.models.autoencoders.components import resnet50_encoder, resnet50_decoder


class LabelDecoder(nn.Module):
    """ 
    Simple module that takes the latent vector as input and outputes logits of each class.
    Can be extended with multiple layers or other loss.
    
    """
    def __init__(self, num_labels, latent_dim):
        super().__init__()

        self.num_labels = num_labels
        self.latent_dim = latent_dim

        self.linear = nn.Linear(self.latent_dim, self.num_labels)
    def forward(self, z):
        return self.linear(z)

    def loss(self, yhat, obs):
        loss = F.cross_entropy(yhat, obs)
        return loss

class ImageDecoder(nn.Module):
    """ resnet decoder """
    def __init__(
        self, 
        latent_dim: int, 
        input_height: int,
        first_conv: bool = False,
        maxpool1: bool = False,
        model_type: str = 'resnet18'
        ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_decoders = {
            'resnet18': resnet18_decoder,
            'resnet50': resnet50_decoder
        }
        self.decoder = valid_decoders[model_type](self.latent_dim, self.input_height, first_conv, maxpool1)
    
    def forward(self, z):
        return self.decoder(z)
    
    def loss(self, yhat, obs):
        loss = F.mse_loss(yhat, obs)
        return loss

class ImageEncoder(nn.Module):
    """ resnet encoder """
    def __init__(
        self, 
        latent_dim: int, 
        enc_out_dim: int = 512,
        first_conv: bool = False,
        maxpool1: bool = False,
        model_type: str = 'resnet18'
        ):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_out_dim = enc_out_dim

        valid_encoders = {
            'resnet18': resnet18_encoder,
            'resnet50': resnet50_encoder
        }
        self.encoder = valid_encoders[model_type](first_conv, maxpool1)
        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)
    
    def forward(self, img):
        x = self.encoder(img)
        mu = self.fc_mu(x)
        var = torch.exp(self.fc_var(x))
        return mu, var

class LabelEncoder(nn.Module):
    """ 
    Simple module that takes the latent vector as input and outputes logits of each class.
    Can be extended with multiple layers or other loss.
    
    """
    def __init__(self, num_labels, latent_dim):
        super().__init__()

        self.num_labels = num_labels
        self.latent_dim = latent_dim

        self.classemb_mu = nn.Embedding(self.num_labels, embedding_dim=self.latent_dim)
        self.classemb_var = nn.Embedding(self.num_labels, embedding_dim=self.latent_dim)
    
    def forward(self, label):
        mu = self.classemb_mu(label)
        var = torch.exp(self.classemb_var(label))
        return mu, var

import transformers
class TextEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(TextEncoder, self).__init__()
        self.transformer = transformers.BertForSequenceClassification.from_pretrained(
                            "bert-base-multilingual-uncased",
                            num_labels = 200,
                            output_attentions = False,
                            output_hidden_states = False
                        )
        self.drop = nn.Dropout(0.1)
        self.mu_fc = nn.Linear(self.transformer.classifier.out_features, latent_dim)
        self.var_fc = nn.Linear(self.transformer.classifier.out_features, latent_dim)

    def forward(self, text):
        logits = self.transformer(text)[0]
        mu = self.mu_fc(logits)
        var = torch.exp(self.var_fc(logits))
        return mu, var

class VAE(pl.LightningModule):
    """
    Modular VAE where you can add multiple generators/discriminators.
    """

    def __init__(
        self,
        input_height: int,
        num_labels= int,
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

        # Define what encoders to use:
        self.encoders_func = nn.ModuleDict({})
        self.encoders_func['image'] = ImageEncoder(latent_dim=latent_dim)
        self.encoders_func['text'] = TextEncoder(latent_dim=latent_dim)
        #self.encoders_func['label'] = LabelEncoder(num_labels=num_labels, latent_dim=latent_dim)
        
        # Define the decoders:
        self.decoders_func = nn.ModuleDict({})
        self.decoders_func['image'] = ImageDecoder(latent_dim=latent_dim, input_height=input_height)
        #self.decoders_func['label'] = LabelDecoder(num_labels=num_labels, latent_dim=latent_dim)

        # Weigh the encoders:
        self.enc2idx = {name : i for i, name in enumerate(self.encoders_func.keys())}
        self.register_parameter("weight_enc",nn.Parameter(torch.ones((len(self.enc2idx))) ))

    def encoder(self, batch):
        # switch to dict batch:
        one_var = next(iter(batch.values()))
        batch_size = len(one_var)
        mu_tot = torch.zeros( (batch_size, self.latent_dim)).to(self.device)
        var_tot = torch.zeros( (batch_size, self.latent_dim)).to(self.device)
        weight_tot = 1e-10

        # Encode all data attributes and add as indep gaussians:
        for key, x in batch.items():
            mu, var = self.encoders_func[key](x)
            w = self.weight_enc[self.enc2idx[key]]
            mu_tot += mu*w
            var_tot += var* w**2
            weight_tot += w
        
        mu_tot = mu_tot/weight_tot
        std_tot = (var_tot/weight_tot).sqrt()
        out = self.sample(mu_tot, std_tot)
        return out

    def decoder(self, z):
        """ Given a latent vector z, generate all data types."""
        data = {}
        for key, decoder in self.decoders_func.items():
            data[key] = decoder(z)
        return data

    def forward(self, batch):
        out = self.encoder(batch)

        # Generate data (decode)
        gen_data = self.decoder(out['z_sample'])
        # Add to output dict
        for key, val in gen_data.items():
            out[key] = val
        return out

    def sample(self, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return {'dist_prior' : p, 'dist_post' : q, 'z_sample' :z}

    def step(self, batch, batch_idx):
        out = self.forward(batch)
        loss_components = {}
        
        # Likelihood loss:
        for key, dec in self.decoders_func.items():
            loss_components[f"loss_{key}"] = dec.loss(out[key], batch[key])
        
        # KL Loss:
        log_qz = out['dist_post'].log_prob(out['z_sample'])
        log_pz = out['dist_prior'].log_prob(out['z_sample'])
        kl = log_qz - log_pz
        loss_components['kl'] = kl.mean()
        
        # Sum the losses:
        loss = 0
        for key, val in loss_components.items():
            loss += val

        logs = loss_components
        logs['loss'] = loss
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        self.log("param/weight", value=self.weight_enc.abs().mean())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val/{k}": v for k, v in logs.items()})

        return loss

    def configure_optimizers(self):
        #return torch.optim.SGD(self.parameters(), lr = self.lr)
        #return OptimizerSGHMC(net=self,alpha = self.lr, nu=0.9, sgmcmc=False)
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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