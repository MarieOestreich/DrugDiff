### NOTE: This is work from Eckmann et al. ! 
### Github: https://github.com/Rose-STL-Lab/LIMO/tree/main
### Paper: https://proceedings.mlr.press/v162/eckmann22a/eckmann22a.pdf

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class VAE(pl.LightningModule):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(max_len * embedding_dim, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, max_len * vocab_len))
        
    def encode(self, x):
        x = self.encoder(self.embedding(x).view((len(x), -1))).view((-1, 2, self.latent_dim))
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var
    
    def decode(self, x):
        return F.log_softmax(self.decoder(x).view((-1, self.max_len, self.vocab_len)), dim=2).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, x):
        z, mu, log_var = self.encode(x)
        return self.decode(z), z, mu, log_var
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return {'optimizer': optimizer}
    
    def loss_function(self, pred, target, mu, log_var, batch_size, p):
        nll = F.nll_loss(pred, target)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (batch_size * pred.shape[1])
        return (1 - p) * nll + p * kld, nll, kld
    
    def training_step(self, train_batch, batch_idx):
        out, z, mu, log_var = self(train_batch)
        p = 0.1
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), train_batch.flatten(), mu, log_var, len(train_batch), p)
        self.log('train_loss', loss)
        self.log('train_nll', nll)
        self.log('train_kld', kld)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var = self(val_batch)
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), val_batch.flatten(), mu, log_var, len(val_batch), 0.5)
        self.log('val_loss', loss)
        self.log('val_nll', nll)
        self.log('val_kld', kld)
        self.log('val_mu', torch.mean(mu))
        self.log('val_logvar', torch.mean(log_var))
        return loss

class PropertyPredictor(pl.LightningModule):
    def __init__(self, in_dim, learning_rate=0.001): # lr was 0.001
        super(PropertyPredictor, self).__init__()
        self.learning_rate = learning_rate
        self.fc = nn.Sequential(nn.Linear(in_dim, 1000),
                                nn.ReLU(),
                                nn.Linear(1000, 1000),
                                nn.ReLU(),
                                nn.Linear(1000, 1))
        
        
    def forward(self, x):
        return self.fc(x)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def loss_function(self, pred, real):
        return F.mse_loss(pred, real)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

# class DiscretePropertyPredictor(pl.LightningModule):
#     def __init__(self, in_dim, num_classes, learning_rate=0.001):
#         super(DiscretePropertyPredictor, self).__init__()
#         self.learning_rate = learning_rate
#         self.fc = nn.Sequential(nn.Linear(in_dim, 1000),
#                                 nn.ReLU(),
#                                 nn.Linear(1000, 1000),
#                                 nn.ReLU(),
#                                 nn.Linear(1000, num_classes))
        
#     def forward(self, x):
#         return self.fc(x)
    
#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.learning_rate)
    
#     def loss_function(self, pred, real):
#         return F.binary_cross_entropy(torch.sigmoid(pred), real)
    
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         out = self(x)
#         loss = self.loss_function(out, y)
#         self.log('train_loss', loss)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         out = self(x)
#         loss = self.loss_function(out, y)
#         self.log('val_loss', loss)
#         return loss


# class GenePropertyPredictor(pl.LightningModule):
#     def __init__(self, mol_dim, gene_dim, learning_rate=0.001, loss = 'mse'):
#         super(GenePropertyPredictor, self).__init__()
#         self.learning_rate = learning_rate
#         self.mol_dim = mol_dim
#         self.mol_fc = nn.Sequential(nn.Linear(mol_dim, 512), # 1000
#                                 nn.ReLU(),
#                                 nn.Linear(512, 256)) #1000, 500
#                                 #nn.ReLU(),
#                                 #nn.Linear(1000, 1))
#         self.gene_fc = nn.Sequential(nn.Linear(gene_dim, 512),
#                                 nn.ReLU(),
#                                 nn.Linear(512, 256))
#                                 #nn.ReLU(),
#                                 #nn.Linear(1000, 1))


#         self.cls = nn.Sequential(nn.ReLU(),
#                                 nn.Linear(512, 256), # 1000, 1000
#                                 nn.ReLU(),
#                                 nn.Linear(256, 2))

#         self.sig = nn.Sigmoid()

#         self.loss = loss
#         self.kl_loss = nn.KLDivLoss(reduction="batchmean")

#     def forward(self, genex, one_hot):
#         # if len(x.shape) > 2:
#         #     x= x.squeeze().float()
#         # else:
#         #     x = x.float()
#         # x1 = x[:, :self.mol_dim]
#         # x2 = x[:, self.mol_dim:]
#         one_hot = one_hot.float()
#         genex = genex.float()
#         mol_emb = self.mol_fc(one_hot)
#         gene_emb = self.gene_fc(genex)
#         com_emb = torch.cat([mol_emb, gene_emb], dim=1)

#         return F.log_softmax(self.cls(com_emb), dim=1)
    
#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.learning_rate)
    
#     def loss_function(self, pred, real):
#         if self.loss == 'mse':
#             return F.mse_loss(pred, real)
#         else:
#             # return F.binary_cross_entropy(pred, real) # needs sigmoid output (see self.cls)
#             # pred = torch.log(pred)
#             target = torch.hstack((1-real.reshape(-1,1), real.reshape(-1,1)))
#             return self.kl_loss(pred, target)
        
#     def training_step(self, batch, batch_idx):
#         genex, onehot, y = batch
#         out = self(genex, onehot)
#         loss = self.loss_function(out, y)
#         self.log('train_loss', loss)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         genex, onehot, y = batch
#         out = self(genex, onehot)
#         loss = self.loss_function(out, y)
#         self.log('val_loss', loss)
#         return loss


# class GenePropertyPredictor(pl.LightningModule):
#     def __init__(self, mol_dim, gene_dim, learning_rate=0.001, loss = 'mse'):
#         super(GenePropertyPredictor, self).__init__()
#         self.learning_rate = learning_rate
#         self.mol_dim = mol_dim
#         self.mol_fc = nn.Sequential(nn.Linear(mol_dim, 512), # 1000
#                                 nn.ReLU(),
#                                 nn.Linear(512, 256)) #1000, 500
#                                 #nn.ReLU(),
#                                 #nn.Linear(1000, 1))
#         self.gene_fc = nn.Sequential(nn.Linear(gene_dim, 512),
#                                 nn.ReLU(),
#                                 nn.Linear(512, 256))
#                                 #nn.ReLU(),
#                                 #nn.Linear(1000, 1))


#         self.cls = nn.Sequential(nn.ReLU(),
#                                 nn.Linear(512, 256), # 1000, 1000
#                                 nn.ReLU(),
#                                 nn.Linear(256, 2))

#         self.sig = nn.Sigmoid()
#         self.temp = 10

#         self.loss = loss
#         self.kl_loss = nn.KLDivLoss(reduction="batchmean")

#     def forward(self, x):
#         if len(x.shape) > 2:
#             x= x.squeeze().float()
#         else:
#             x = x.float()
#         x1 = x[:, :self.mol_dim]
#         x2 = x[:, self.mol_dim:]

#         mol_emb = self.mol_fc(x1)
#         gene_emb = self.gene_fc(x2)
#         com_emb = torch.cat([mol_emb, gene_emb], dim=1)

#         # return F.log_softmax(self.cls(com_emb), dim=1)
#         return self.cls(com_emb)
    
#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.learning_rate)
    
#     def loss_function(self, pred, real):
#         if self.loss == 'mse':
#             return F.mse_loss(pred, real)
#         else:
#             sig_pred = self.sig(pred)
#             target = torch.hstack((1-real.reshape(-1,1), real.reshape(-1,1)))
            
#             target_bin = real
#             target_bin[target_bin < 0.5] = 0
#             target_bin[target_bin > 0.5] = 1
#             target_bin = torch.hstack((1-target_bin.reshape(-1,1), target_bin.reshape(-1,1)))

#             bce_loss = F.binary_cross_entropy(sig_pred, target_bin) # needs sigmoid output (see self.cls)
#             # return F.binary_cross_entropy(pred, real) # needs sigmoid output (see self.cls)
#             # pred = torch.log(pred)
#             log_pred = F.log_softmax(pred, dim=1)
#             kll = self.kl_loss(log_pred/self.temp, target/self.temp) * self.temp * self.temp
#             return 0.1*bce_loss + 0.9*kll
            

    
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         out = self(x)
#         loss = self.loss_function(out, y)
#         self.log('train_loss', loss)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         out = self(x)
#         loss = self.loss_function(out, y)
#         self.log('val_loss', loss)
#         return loss

# class genexVAE(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(978, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024 * 2)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 978)
#         )

#         self.loss = nn.MSELoss()
    
#     def loss_function(self, pred, target, mu, log_var):
#         mse = self.loss(pred, target)
#         kld = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = 1)).mean(dim =0)    
#         return mse + kld
        
#     def forward(self, x):
#         z, mu, log_var = self.encode(x)
#         x_hat = self.decoder(z)
#         return x_hat, z, mu, log_var
    
#     def encode(self, x):
#         z = self.encoder(x).view((-1, 2, 1024))
#         mu, log_var = z[:, 0, :], z[:, 1, :]
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std, mu, log_var
    
#     def training_step(self, batch, batch_idx):
#         x = batch.unsqueeze(1)
#         x_hat, z, mu, log_var = self(x)
#         loss = self.loss_function(x_hat, x, mu, log_var)
#         self.log('train/loss', loss)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x = batch.unsqueeze(1)
#         x_hat, z, mu, log_var = self(x)
#         loss = self.loss_function(x_hat, x, mu, log_var)
#         self.log('val/loss', loss)
#         return loss
    
#     def test_step(self, batch, batch_idx):
#         x = batch.unsqueeze(1)
#         x_hat, z, mu, log_var = self(x)
#         loss = self.loss_function(x_hat, x, mu, log_var)
#         self.log('test/loss', loss)
#         return loss
    
#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=0.0001)
#         return {'optimizer': optimizer}