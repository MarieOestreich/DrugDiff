import os
import sys
# Add the parent directory to the system path
FILE_DIR = __file__.split('train_predictor_target_affinity.py')[0] + ".."
sys.path.append(FILE_DIR)

from pathlib import Path
import torch
from torch import nn
import json
import pickle
import selfies as sf
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.stats import linregress, pearsonr
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.models.components.vae import PropertyPredictor


# Functions from LIMO to translate SMILES into one-hot encodings:

def smiles_to_indices(smiles):
    encoding = [symbol_to_idx[symbol] for symbol in sf.split_selfies(sf.encoder(smiles))]
    return torch.tensor(encoding + [symbol_to_idx['[nop]'] for i in range(max_len - len(encoding))])


def smiles_to_one_hot(smiles):
    out = torch.zeros((max_len, len(symbol_to_idx)))
    for i, index in enumerate(smiles_to_indices(smiles)):
        out[i][index] = 1
    return out.flatten()

# if you used docking/scripts/dlg2aff.py during the docking precedure to extract your affinities, then you should have
# a .csv file with smiles and their binding affinities (in a column labelled "affinity").
PROPERTY = 'affinity'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'You are running on the following device:    {device}')

# We will use the same symbol encodings as the VAE and Diffusion Model are using to encode the molecules for training
# the property predictor. This is essential because during guidance, the predictor will receive one-hot encoded molecules
# from the VAE's decoder and their symbol arrangement must match the one the predictor was trained on.
symbol_to_idx = pickle.load(open(FILE_DIR + '../data/zinc250k/symbol_to_idx.pickle', "rb"))
idx_to_symbol = pickle.load(open(FILE_DIR + '../data/zinc250k/idx_to_symbol.pickle', "rb"))
max_len = pickle.load(open(FILE_DIR + '../data/zinc250k/dataset_max_len.pickle', "rb"))



# data module to be used by the predictor model
class PropDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super(PropDataModule, self).__init__()
        self.batch_size = batch_size
        self.x = torch.load(FILE_DIR + '../data/zinc250k/esr1_one_hots_and_target_energies/x.pt')
        self.y = torch.load(FILE_DIR + '../data/zinc250k/esr1_one_hots_and_target_energies/y.pt')
        self.dataset = TensorDataset(self.x, self.y)
        self.train_data, self.val_data = random_split(self.dataset, [int(round(len(self.dataset) * 0.9)), len(self.dataset) - int(round(len(self.dataset) * 0.9))])
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, drop_last=True)
    
model  = PropertyPredictor(in_dim=max_len * len(symbol_to_idx))
batch_size = 2048 # batch size for predictor training
max_num_epochs = 5 # number of epochs to train for

# specify a path to save model file, and plots
save_dir = f"{FILE_DIR}../checkpoints/property_predictors/{PROPERTY}/"
# create the path if it does not yet exist
Path(save_dir).mkdir(parents=True, exist_ok=True)

dm = PropDataModule(batch_size) 
checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename='best', monitor='val_loss')

# train the predictor
trainer = pl.Trainer(callbacks=[checkpoint_callback], accelerator='gpu', devices=4, max_epochs=max_num_epochs, enable_checkpointing=True, 
                    logger=pl.loggers.CSVLogger('logs'),log_every_n_steps=1,
                    gradient_clip_val=0.5)

trainer.fit(model, dm)