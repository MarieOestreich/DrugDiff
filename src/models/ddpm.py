"""
adapted from 
https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py
-- gracias
"""

"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import os
import torch
import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from functools import partial
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw
from src.utils.util import default
from src.utils.diff_utils import make_beta_schedule, extract_into_tensor, noise_like
from src.models.components.vae import VAE
from src.models.components.denseddpm import  DenseDDPM
from tqdm import tqdm
from src.models.components.evaluation import Evaluator
import numpy as np
import matplotlib.pyplot as plt
from statistics import median
import sys
import datetime

import pathlib
FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
FILE_PATH = "/".join(FILE_PATH.split("\\"))
print(FILE_PATH)
REPO_PATH = FILE_PATH.split('DrugDiff')[0] + 'DrugDiff'
sys.path.append(REPO_PATH)
print(REPO_PATH)

class LatentDiffusion(pl.LightningModule):
    """main class"""
    def __init__(self,
                 input_dim=1024,
                 mlp_dims=2048,
                 timesteps=1000,
                 dataset= 'zinc250k',
                 beta_schedule="linear",
                 clip_denoised=False,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 parameterization="eps",
                 lr_rate=0.0001,
                 v_posterior=0.,
                 input_type="vae", #ae
                 vae_ckpt=None,
                 uncond_dm_ckpt = None,
       
                 *args, **kwargs):
        
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.lr_rate = lr_rate
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        self.input_dim = input_dim
        self.uncond_dm_ckpt = uncond_dm_ckpt

        self.model = DenseDDPM(input_dim=input_dim, mlp_dims=mlp_dims)

        self.dataset=dataset

        # determines if loss contributions will be scaled or not during sampling
        self.scale_losses_sampling = False


        with open(REPO_PATH + '/data/zinc250k/symbol_to_idx.pickle', "rb") as input_file:
            self.symbol_to_idx = pickle.load(input_file)
        with open(REPO_PATH + '/data/zinc250k/idx_to_symbol.pickle', "rb") as input_file:
            self.idx_to_symbol = pickle.load(input_file)
        with open(REPO_PATH + '/data/zinc250k/dataset_max_len.pickle', "rb") as input_file:
            self.max_len = pickle.load(input_file)

        self.vae = VAE(max_len=self.max_len, vocab_len=len(self.symbol_to_idx), latent_dim=input_dim, embedding_dim=64) 
        
        if self.uncond_dm_ckpt is not None:
            print ('========== USING DDPM CHECKPOINT ===========')

            model_ckpt = torch.load(self.uncond_dm_ckpt,
                    map_location=torch.device(self.device))
                    
            sd = self.state_dict()

            for k in sd.keys():
                sd[k] = model_ckpt['state_dict'][k]
                
            self.load_state_dict(sd)
            self.model.eval()
        else:
            print(f' =============================== VAE CKPT: {vae_ckpt} ================================')
            self.vae.load_state_dict(torch.load(vae_ckpt))

        self.vae.eval()

        for param in self.vae.parameters():
            param.requires_grad = False

        self.prop_weight_values = []
        self.classifiers = {}

        self.geneX = None   
                
        self.v_posterior = v_posterior
        self.input_type = input_type


        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)    

     
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
     
        self.vae.to(device)
        self.vae.encoder.to(device)
        self.vae.decoder.to(device)

        print("***** INITIALIZATION IS FINISHED ******* ")
        

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):

        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def shared_step(self, x, **kwargs):
            if self.input_type =='selfies':
                x, _, _ = self.vae.encode(x)

            loss = self(x)
            return loss

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def training_step(self, batch, batch_idx):

   
        batch, mu, log_var = self.vae.encode(batch)
        batch = batch.detach().squeeze()

        loss = self.shared_step(batch)
        self.log("train/loss",loss, prog_bar=True,logger=True, on_step=False, on_epoch=True)
        return loss
     

    def validation_step(self, batch, batch_idx):
        
        batch, mu, log_var = self.vae.encode(batch)
        batch = batch.detach().squeeze()

        loss = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)


    def apply_model(self, x_noisy, t):

        x_recon = self.model(x = x_noisy, t = t)
  
        return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        
    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        model_output = self.apply_model(x_noisy, t)

        target = noise

        loss = torch.nn.functional.mse_loss(model_output, target)

        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, x, t, clip_denoised: bool,
                        return_x0=True):
        
  
        model_out = self.apply_model(x, t)
        
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)  

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
       
        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def min_max_normalize(self, tensor):
        """
        Applies element wise Min-max scaling to a PyTorch tensor.

        Args:
            tensor (torch.Tensor): The input PyTorch tensor to be scaled.

        Returns:
            torch.Tensor: A new tensor with the same shape as the input tensor, but with its elements
                          scaled to the range [0, 1] using the formula:
                          scaled_value = (value - min_value) / (max_value - min_value),
                          where min_value is the minimum element in the input tensor, and
                          max_value is the maximum element in the input tensor.

        Example:
            input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
            scaled_tensor = min_max_normalize(input_tensor)
            scaled_tensor would be: tensor([0.0000, 0.3333, 0.6667, 1.0000])
        """
        min_value = tensor.min()
        max_value = tensor.max()

        normalized_tensor = (tensor - min_value) / (max_value - min_value)

        return normalized_tensor


    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, repeat_noise=False,
                 return_x0=True, temperature=1., noise_dropout=0., classifier_scale=1.,
                 start_idx = 0, end_idx = -1, log_mols_every_n = None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised,
                                       return_x0=return_x0)
        if return_x0:
            model_mean, model_variance, model_log_variance, x0 = outputs
        else:
            model_mean, model_variance, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        

        if end_idx == -1:
            end_idx = x0.shape[0]

        if len(self.classifiers) > 0:
            with torch.enable_grad():
                loss = 0
                x_in = x0.detach().requires_grad_(True)
                sampled_mols = torch.exp(self.vae.decode(x_in[start_idx:end_idx, :].to(self.device))) 

                if log_mols_every_n is not None and ((int(t[0]+1) % log_mols_every_n == 0) or int(t[0]) == 0):
                    # sampled_mols to smiles
                    sampled_smiles = [self.one_hot_to_smiles(hot) for hot in sampled_mols]
                    # save as csv named after mols-date-time:
                    sampled_smiles_df = pd.DataFrame(sampled_smiles, columns = ['smiles'])
                    pathlib.Path(REPO_PATH + "/mol_over_t/" + datetime.datetime.now().strftime("%b-%d-%Y-%H") + '/').mkdir(parents=True, exist_ok=True)
                    sampled_smiles_df.to_csv(REPO_PATH + "/mol_over_t/" + datetime.datetime.now().strftime("%b-%d-%Y-%H") + f'/mols-{str(int(t[0]))}-g-{str(int(classifier_scale))}')


                if self.substruct_smile:
                   
                    loss += -1 * torch.sum(((sampled_mols - self.orig_x_repeated.clone().detach().to(device)) * self.mask_repeated.to(device)) ** 2)
                
                for prop in self.classifiers.keys():
                    predictor_model = self.classifiers[prop].to(device)
                    predictor_model.eval() 
                    if prop == 'L1000':
                        gene_in = self.geneX[start_idx:end_idx, :].detach().requires_grad_(False)
                        # mols_comb = torch.cat([sampled_mols.to(device), gene_in.to(device)],dim=1)
                        # log_probs = predictor_model(mols_comb.to(device))
                        out = predictor_model(gene_in.to(self.device), sampled_mols.to(self.device)).squeeze()
                        # out = log_probs[:,1]

                        # here out is not Min-max scaled, because it wasn't tested for L1000, but probably it can be applied here too
                        # out = self.min_max_normalize(tensor=out)
                        if self.scale_losses_sampling:
                            property_contribution_to_loss = torch.sum(self.min_max_normalize(
                                tensor=out))
                        else:
                            property_contribution_to_loss = torch.sum(out)
                        # in this version, the sign of the weight determines the target value direction
                        prop_loss = property_contribution_to_loss * self.prop_weights[
                            prop]  # prop_loss is the actual value that will contribute to loss after scaling and weighting
                        loss += prop_loss

                    else:
                        out = predictor_model(sampled_mols)

                        # 'out' is the tensor we want to scale, it contains outputs of different property predictors that are on different scales
                        if self.scale_losses_sampling:
                            property_contribution_to_loss = torch.sum(self.min_max_normalize(
                                tensor=out))
                        else:
                            property_contribution_to_loss = torch.sum(out)
                        # in this version, the sign of the weight determines the target value direction
                        prop_loss = property_contribution_to_loss * self.prop_weights[
                            prop]  # prop_loss is the actual value that will contribute to loss after scaling and weighting
                        loss += prop_loss

                gradient = torch.autograd.grad(loss, x_in)[0] * classifier_scale 
            model_mean = model_mean + model_variance * gradient

            
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_T=None, verbose=True, timesteps=None, 
                      mask=None, x0=None, start_T=None, classifier_scale=1.,
                 start_idx = 0, end_idx = -1, log_mols_every_n = None, stop_after_t = None):
        

       
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            latent = torch.randn(shape, device=device)
        else:
            latent = x_T
            latent = latent.to(device)

        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        
        if stop_after_t is not None:
            iterator = tqdm(reversed(range(timesteps-stop_after_t, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
                range(timesteps-stop_after_t, timesteps))
        else:
            iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
                range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            # print(f'ts (expecting to be tensor of same number repeated): {ts}')

            latent, _ = self.p_sample(latent, ts, clip_denoised=self.clip_denoised, classifier_scale=classifier_scale,
                 start_idx = 0, end_idx = -1, log_mols_every_n = log_mols_every_n)
            if mask is not None:
                latent_orig = self.q_sample(x0, ts)
                latent = latent_orig * mask + (1. - mask) * latent

            
        return latent

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    @torch.no_grad()
    def sample(self, 
               batch_size=10000, 
               x_T=None,
               verbose=True, 
               timesteps=None,
               mask=None, 
               x0=None, 
               start_T=None, 
               sample_dim=None, 
               classifier_scale= 1., 
               start_idx = 0, end_idx = -1, 
               log_mols_every_n = None, 
               stop_after_t = None, 
               **kwargs):
        if sample_dim is None:
            shape = (batch_size, 1024)
        else:
            shape = (batch_size, sample_dim)
      
        return self.p_sample_loop(shape, x_T=x_T, verbose=verbose, timesteps=timesteps,mask=mask, x0=x0, start_T=start_T, classifier_scale=classifier_scale,
                 start_idx = 0, end_idx = -1, log_mols_every_n = log_mols_every_n, stop_after_t = stop_after_t)

    def configure_optimizers(self):
        lr = self.lr_rate
        params = list(self.model.parameters())
       
        opt = torch.optim.AdamW(params, lr=lr)
        
        return opt


    def smiles_to_indices(self, smiles):

        encoding = [self.symbol_to_idx[symbol] for symbol in sf.split_selfies(sf.encoder(smiles))]
        return torch.tensor(encoding + [self.symbol_to_idx['[nop]'] for i in range(self.max_len - len(encoding))])


    def smiles_to_one_hot(self, smiles):
        out = torch.zeros((self.max_len, len(self.symbol_to_idx)))
        for i, index in enumerate(self.smiles_to_indices(smiles)):
            out[i][index] = 1
        return out.flatten()
    
    def substruct_to_midpos(self, smiles):
        out = torch.zeros((self.max_len, len(self.symbol_to_idx)))
        smiles_idx = [self.symbol_to_idx[symbol] for symbol in sf.split_selfies(sf.encoder(smiles))] # indeces without nop
        len_smiles = len(smiles_idx)
        if len_smiles == self.max_len:
            start_idx = 0
        else:
            start_idx = self.max_len//2 - len_smiles//2
        for i, index in enumerate(smiles_idx):
            out[i+start_idx][index] = 1
        return out
        
    def place_substruct(self, substruct, sampled):
        shape = sampled.shape
        sampled = torch.reshape(sampled, (sampled.shape[0],self.max_len, len(self.symbol_to_idx)))
        substruct = self.substruct_to_midpos(substruct).repeat(sampled.shape[0], 1, 1)
        for i in range(substruct.shape[1]):
            if not torch.sum(substruct[:, i,:]):
                substruct[:, i] = sampled[:, i]
        substruct = substruct.reshape(shape[0], -1)
        return substruct
        

    def smiles_to_z(self, smiles, vae, device):
        zs = torch.zeros((len(smiles), 1024), device=device)
        for i, smile in enumerate(tqdm(smiles)):

            z = vae.encode(self.smiles_to_indices(smile).unsqueeze(0).to(device))[0].detach().requires_grad_(True)
            
            zs[i] = z.detach()
        return zs


    def one_hot_to_selfies(self, hot):
        return ''.join([self.idx_to_symbol[idx.item()] for idx in hot.view((self.max_len, -1)).argmax(1)]).replace(' ', '')

    def one_hot_to_smiles(self, hot):
        return sf.decoder(self.one_hot_to_selfies(hot))


    @torch.no_grad()
    def apply_guidance(self,
                        classifiers,
                        properties,
                        weights,
                        classifier_scales,
                        sample_num,
                        exp_name,
                        filter = None,
                        margins = None,
                        use_anchor=False,
                        anchor_smile = "",
                        substruct_smile = None,
                        scale_losses=True,
                        log_mols_every_n = None,
                        start_T = None,
                        stop_after_t = None,
                        geneX = None):  # if scale_losses is True, losses will be scaled during sampling

        pl.seed_everything(215687) # 12345
        self.load_classifiers(properties, weights, classifiers)

        evaluator = Evaluator(self.max_len, self.symbol_to_idx)
        print(f'++++ PROPRETIES USED: {properties}')
        print(f'++++ WEIGHTS USED: {weights}')
        
        self.model.eval()
        self.sample_num = sample_num
        self.use_anchor = use_anchor
        self.substruct_smile = substruct_smile
        self.geneX = geneX

        if scale_losses:
            print(f"++++ Using Min-max scaling for losses during sampling")
            self.scale_losses_sampling = True
        else:
            print(f"++++ Not scaling losses during sampling")
            self.scale_losses_sampling = False



        if self.substruct_smile:
            self.anchor_smile = anchor_smile
            self.orig_z = self.smiles_to_z([self.anchor_smile], self.vae, self.device)

            self.orig_x = torch.exp(self.vae.decode(self.orig_z))

            self.substruct = Chem.MolFromSmiles(self.substruct_smile) # put smiles of the substructure you wish to keep constant here
            selfies = list(sf.split_selfies(sf.encoder(self.anchor_smile)))
            self.mask = torch.zeros_like(self.orig_x)
            self.pos_mask = torch.zeros_like(self.orig_x)
            for i in range(len(selfies)):
                for j in range(len(self.idx_to_symbol)):
                    changed = selfies.copy()
                    changed[i] = self.idx_to_symbol[j]
                    m = Chem.MolFromSmiles(sf.decoder(''.join(changed)))
                    if not m.HasSubstructMatch(self.substruct):
                        self.mask[0][i * len(self.idx_to_symbol) + j] = 1
                    else:
                        self.pos_mask[0][i * len(self.idx_to_symbol) + j] = 1

            self.orig_z_repeated = self.orig_z.repeat(self.sample_num,1)
            self.orig_x_repeated = self.orig_x.repeat(self.sample_num,1)
            self.mask_repeated = self.mask.repeat(self.sample_num,1)
        else:
            self.substruct = None

    #########################################

        sampled_all = pd.DataFrame()
        mols_all = torch.Tensor().to(self.device)
        for cs in classifier_scales:
            
            start = 0
            while self.input_dim > start:
                sampled_cs = pd.DataFrame()
                if self.input_dim > start + 10000:
                    end = start + 10000
                else:
                    end = -1

                sampled_latents = self.sample(sample_dim=self.input_dim, batch_size=sample_num, classifier_scale=cs, start_idx=start, end_idx=end, 
                                              log_mols_every_n = log_mols_every_n, start_T=start_T, stop_after_t = stop_after_t)
                if end == -1:
                    sampled_mols = torch.exp(self.vae.decode(sampled_latents[start:, :].to(self.device)))
                else:
                    sampled_mols = torch.exp(self.vae.decode(sampled_latents[start:end+1, :].to(self.device)))
 
                mols_all = torch.cat((mols_all, sampled_mols))

                sampled_smiles = [self.one_hot_to_smiles(hot) for hot in sampled_mols]
                sampled_smiles_canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles = True) for s in  sampled_smiles]

                sampled_cs["smiles"] = sampled_smiles
                sampled_cs["canonical_smiles"] = sampled_smiles_canonical
                sampled_cs["guidance"] = [cs] * len(sampled_smiles)

                if self.geneX is not None:
                    if end == -1:
                        genex_tmp = self.geneX[start:, :]
                    else:
                        genex_tmp = self.geneX[start:end, :]

                else:
                    genex_tmp = None
             
                    
                mol_props = evaluator.computeProperties(sampled_smiles)
                mol_props_pred = evaluator.precictProperties(sampled_smiles, sampled_mols, self.classifiers, genex = genex_tmp)
                

                for prop in self.classifiers.keys():
                        mol_props_pred[f"{prop+'_pred'}_{self.prop_weights[prop]}"] = mol_props_pred[prop+'_pred'].values
              
                mol_props = pd.concat([mol_props, mol_props_pred], axis=1)
                sampled_cs = pd.concat([sampled_cs, mol_props], axis=1)

                sampled_all = pd.concat([sampled_all, sampled_cs])

                start += 10000

            tmp = sampled_all[sampled_all["guidance"] == cs]
            tmp = tmp.drop_duplicates("canonical_smiles")

            for prop in self.classifiers.keys():
                prop_ = prop+"_pred"
                print(f"================= Min unfiltered {prop}: {min(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
                print(f"====================== Median unfiltered {prop}: {median(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
                print(f"=========================== Max unfiltered {prop}: {max(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
            if filter:
                tmp = filter(tmp)
                for prop in self.classifiers.keys():
                    
                    print(f"================= Min filtered {prop}: {min(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
                    print(f"====================== Median filtered {prop}: {median(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
                    print(f"=========================== Max filtered {prop}: {max(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
           
            print(f'/////// CALCULATED DDPM CLASSIFIER {cs}///////')
        if not os.path.exists(REPO_PATH + '/outputs/'):
            os.makedirs(REPO_PATH + '/outputs/')

        sampled_all.to_csv(exp_name + '.csv')
        torch.save(mols_all, exp_name + '.pt')
        

    def load_classifiers(self, prop_names, prop_weights, clf_list):
        self.classifiers = {}
        self.prop_weights = {}

        for i, prop in enumerate(prop_names):
            self.classifiers[prop] = clf_list[i]
            self.prop_weights[prop] = prop_weights[i]

    
        
