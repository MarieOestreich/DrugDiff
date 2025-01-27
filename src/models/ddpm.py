"""
adapted from 
https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py
-- gracias
"""

# Import necessary libraries
import os
import torch
import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from functools import partial
import selfies as sf
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
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

# Get the current file path and convert it to a Unix-style path for consistency
FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
FILE_PATH = "/".join(FILE_PATH.split("\\"))


class LatentDiffusion(pl.LightningModule):
    """main class"""
    def __init__(self,
                 input_dim=1024,
                 mlp_dims=2048,
                 timesteps=1000,
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

        # Validate the parameterization type
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'

        # Initialize model attributes
        self.lr_rate = lr_rate
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        self.input_dim = input_dim
        self.uncond_dm_ckpt = uncond_dm_ckpt

        # Initialize the DenseDDPM model
        self.model = DenseDDPM(input_dim=input_dim, mlp_dims=mlp_dims)

        # Flag to scale loss contributions during sampling
        self.scale_losses_sampling = False

        # load symbol mappings and maximum molecule string length information from trained VAE
        with open(FILE_PATH + '/../../checkpoints/vae/'+'symbol_to_idx.pickle', "rb") as input_file:
            self.symbol_to_idx = pickle.load(input_file)

        with open(FILE_PATH + '/../../checkpoints/vae/'+'idx_to_symbol.pickle', "rb") as input_file:
            self.idx_to_symbol = pickle.load(input_file)

        with open(FILE_PATH + '/../../checkpoints/vae/'+'dataset_max_len.pickle', "rb") as input_file:

            self.max_len = pickle.load(input_file)
   
        # Initialize the VAE model
        self.vae = VAE(max_len=self.max_len, vocab_len=len(self.symbol_to_idx), latent_dim=input_dim, embedding_dim=64) 
    
        # Load pre-trained unconditional diffusion model checkpoint (if provided)
        if self.uncond_dm_ckpt is not None:
            # This is used during generation with or without guidance and includes the VAE checkpoint used when orignally training the diffusion model.
            # Hence, explicit loading of a pre-trained VAE is not necessary
            print ('========== USING DDPM CHECKPOINT ===========')

            model_ckpt = torch.load(self.uncond_dm_ckpt,
                    map_location=torch.device(self.device))
                    
            sd = self.state_dict()

            for k in sd.keys():
                sd[k] = model_ckpt['state_dict'][k]
                
            self.load_state_dict(sd)
            self.model.eval()
        else:
            # Load pre-trained VAE checkpoint needed for training the DrugDiff model
            print(f' =============================== VAE CKPT: {FILE_PATH + "/../.." + vae_ckpt} ================================')
            self.vae.load_state_dict(torch.load(FILE_PATH + '/../..' + vae_ckpt))

        # Set the VAE to evaluation mode and freeze its parameters
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # Initialize empty lists and dictionaries for future use
        self.prop_weight_values = []
        self.classifiers = {}

        self.geneX = None # this is a remnant of experiments with guidance by gene expression values. Consider to remove.
                
        # Set the posterior variance and input type attributes
        self.v_posterior = v_posterior
        self.input_type = input_type

        # Register the beta schedule for the diffusion process
        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)    

        # Determine the device to run on (GPU or CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        print(f'Running on the following device:    {device}')             
     
        # Move the VAE model and its components to the determined device
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

        """
        Shared step function for training and validation.

        This function encodes the input data using the VAE (if necessary) and then calculates the loss.

        Args:
            x: The input data.
            **kwargs: Additional keyword arguments.

        Returns:
            loss: The calculated loss.
        """
        if self.input_type =='selfies':
            x, _, _ = self.vae.encode(x)

        loss = self(x)
        return loss

    def forward(self, x, *args, **kwargs):
        # select a random time step
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        
        # Call the p_losses function to calculate the loss
        return self.p_losses(x, t, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        # Encode the input data using the VAE
        batch, mu, log_var = self.vae.encode(batch)

        # Detach the encoded data
        batch = batch.detach().squeeze()

        # Calculate the loss by calling the shared_step function
        loss = self.shared_step(batch)
        
        # Log the loss to the logger
        self.log("train/loss",loss, prog_bar=True,logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
     

    def validation_step(self, batch, batch_idx):
        
        batch, mu, log_var = self.vae.encode(batch)
        batch = batch.detach().squeeze()

        loss = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        
        batch, mu, log_var = self.vae.encode(batch)
        batch = batch.detach().squeeze()

        loss = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)


    def apply_model(self, x_noisy, t):
        """
        Apply model function.

        This function applies the model to the noisy input data at a given time step.

        Args:
            x_noisy: The noisy input data.
            t: The time step.

        Returns:
            x_recon: The reconstructed data.
        """
        # Apply the model to the noisy input data
        x_recon = self.model(x = x_noisy, t = t)
  
        return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the Q distribution.

        This function samples a noisy version of the input data at time step t using the Q distribution.

        Args:
            x_start: The input data.
            t: The time step.
            noise: The noise to use for sampling. If None, a random noise is generated.

        Returns:
            The sampled noisy data.
        """
        # Generate a random noise if none is provided
        noise = default(noise, lambda: torch.randn_like(x_start))
        # Sample from the Q distribution using the formula for the forward process
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        
    def p_losses(self, x_start, t, noise=None):
        """
        Calculate the loss for the P distribution.

        This function calculates the mean squared error between the predicted noise and the target noise.

        Args:
            x_start: The input data.
            t: The time step.
            noise: The noise to use for sampling. If None, a random noise is generated.

        Returns:
            The calculated loss.
        """
        # Generate a random noise if none is provided
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Sample a noisy version of the input data at time step t using the Q distribution
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Apply the model to the noisy data
        model_output = self.apply_model(x_noisy, t)

        # The target value is the original noise
        target = noise

        # Calculate the mean squared error between the predicted and target noise
        loss = torch.nn.functional.mse_loss(model_output, target)

        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict the start of the sequence from the noisy data.

        This function calculates the predicted start value given the input data at time step t and the noise value.

        Args:
            x_t: The input data at time step t.
            t: The time step.
            noise: The noise value.

        Returns:
            The predicted start value.
        """

        # Calculate the predicted start value using the formula for the reverse process
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, x, t, clip_denoised: bool,
                        return_x0=True):
        
        """
        Calculate the mean and variance of the P distribution.

        This function calculates the mean and variance of the predicted added noise given the input data at time step t.

        Args:
            x: The input data.
            t: The time step.
            clip_denoised: Whether to clip the denoised values to the range [-1, 1].
            return_x0: Whether to also return the predicted start value or just the predicted noise.

        Returns:
            The calculated mean, variance, and log variance of the P distribution. If return_x0 is True, also returns the predicted start value.
        """
        # Apply the model to the input data
        model_out = self.apply_model(x, t)

        # Based on the parameterization ...
        if self.parameterization == "eps":
            # ... predict start value, or
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            # ... predict added noise
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)  

        # Calculate the mean, variance, and log variance of the P distribution using the q_posterior function
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
        """
        Samples from the diffusion process.

        Args:
            x (Tensor): Input tensor.
            t (int): Time step.
            clip_denoised (bool): Whether to clip denoised values. 
            repeat_noise (bool): Whether to repeat noise for full batch. 
            return_x0 (bool): Whether to return the predicted sample at t0. Defaults to True.
            temperature (float): Temperature for sampling. Defaults to 1.
            noise_dropout (float): Dropout rate for noise. Defaults to 0.
            classifier_scale (float): Scale for classifier loss. Defaults to 1.
            start_idx (int): Start index for diffusing and decoding in mini batches (if memory too loww for the whole batch). Defaults to 0.
            end_idx (int): End index for diffusing and decoding in mini batches. Defaults to -1 (i.e. the full batch).
            log_mols_every_n (int): Log molecules every n steps. Defaults to None.

        Returns:
            Tensor or tuple of Tensors: Sampled value(s).
        """

        # Get batch size, shape, and device from input tensor
        b, *_, device = *x.shape, x.device
        # Compute mean and variance of predicted nosie using the diffusion model
        outputs = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised,
                                       return_x0=return_x0)

        # Unpack outputs based on whether x0 is returned
        if return_x0:
            model_mean, model_variance, model_log_variance, x0 = outputs
        else:
            model_mean, model_variance, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature

        # Apply dropout to noise if specified
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # Create a mask to prevent noise from being added at time step 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
        # Adjust end index to full batch if not specified otherwise
        if end_idx == -1:
            end_idx = x0.shape[0]

        # If classifiers are available, compute the classifier loss and update the model mean
        if len(self.classifiers) > 0:
            with torch.enable_grad():
                # Initialize loss to zero
                loss = 0
                # Detach and require gradients for the input tensor
                x_in = x0.detach().requires_grad_(True)
                # Decode and compute sampled molecules using the VAE
                sampled_mols = torch.exp(self.vae.decode(x_in[start_idx:end_idx, :].to(self.device))) 

                # Log molecules every n steps if specified
                if log_mols_every_n is not None and ((int(t[0]+1) % log_mols_every_n == 0) or int(t[0]) == 0):
                    # sampled_mols to smiles
                    sampled_smiles = [self.one_hot_to_smiles(hot) for hot in sampled_mols]
                    # save as csv named after mols-date-time:
                    sampled_smiles_df = pd.DataFrame(sampled_smiles, columns = ['smiles'])


                    ### FIX ###
                    # Remove use of REPO_PATH

                    pathlib.Path(REPO_PATH + "/mol_over_t/" + datetime.datetime.now().strftime("%b-%d-%Y-%H") + '/').mkdir(parents=True, exist_ok=True)
                    sampled_smiles_df.to_csv(REPO_PATH + "/mol_over_t/" + datetime.datetime.now().strftime("%b-%d-%Y-%H") + f'/mols-{str(int(t[0]))}-g-{str(int(classifier_scale))}')

                # Masking parts of latent to fix substructure. Only experimental, to be revisited later.
                # if self.substruct_smile:
                   
                #     loss += -1 * torch.sum(((sampled_mols - self.orig_x_repeated.clone().detach().to(device)) * self.mask_repeated.to(device)) ** 2)
                
                # Iterate over classifiers and compute their losses
                for prop in self.classifiers.keys():
                    predictor_model = self.classifiers[prop].to(device)
                    predictor_model.eval() 
                    
                    # Remnant of experimentation with gene expression-based guidance
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

                    # Compute output using the predictor model
                    else:
                        # Create the tensor we want to scale, it contains outputs of different property predictors that are on different scales
                        out = predictor_model(sampled_mols)

                        # Compute loss and add it to the total loss
                        if self.scale_losses_sampling:
                            property_contribution_to_loss = torch.sum(self.min_max_normalize(
                                tensor=out))
                        else:
                            property_contribution_to_loss = torch.sum(out)

                        # the sign of the weight determines the target value direction
                        prop_loss = property_contribution_to_loss * self.prop_weights[
                            prop]  # prop_loss is the actual value that will contribute to loss after scaling and weighting
                        loss += prop_loss
                # Backpropagate gradients
                gradient = torch.autograd.grad(loss, x_in)[0] * classifier_scale 
            # Update the model mean
            model_mean = model_mean + model_variance * gradient

            
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_T=None, verbose=True, timesteps=None, 
                      mask=None, x0=None, start_T=None, classifier_scale=1.,
                 start_idx = 0, end_idx = -1, log_mols_every_n = None, stop_after_t = None):
        
        """
        Samples from the diffusion model using a loop over the time steps.

        Args:
            shape: Shape of the latents.
            x_T (tensor): Input tensor at time T.
            verbose (bool): Whether to print progress through loop. Defaults to True.
            timesteps (int): Number of time steps to diffuse through. 
            mask (tensor): Mask for masking part of the latent. 
            x0 (tensor): Input tensor at time 0 (not noised). 
            start_T (int): Start time step. 
            classifier_scale (float): Scale factor for classifiers. Defaults to 1.
            start_idx (int): Start index for diffusing and decoding in mini batches (if memory too loww for the whole batch). Defaults to 0.
            end_idx (int): End index for diffusing and decoding in mini batches. Defaults to -1 (i.e. the full batch).
            log_mols_every_n (int): Frequency for logging molecules every n time steps.
            stop_after_t (int): Time step to stop diffusion process.

        Returns:
            Sampled tensor at the final time step.
        """

        # Get device from betas
        device = self.betas.device

        # Initialize batch size and input tensor (i.e. latent to diffuse)
        b = shape[0]
        if x_T is None:
            # If no input tensor, sample a random one
            latent = torch.randn(shape, device=device)
        else:
            # Otherwise, use the given input tensor
            latent = x_T
            latent = latent.to(device)

        # Set time steps and start time step
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        
        # Create an iterator and progress bar for the sampling process
        if stop_after_t is not None:
            iterator = tqdm(reversed(range(timesteps-stop_after_t, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
                range(timesteps-stop_after_t, timesteps))
        else:
            iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
                range(0, timesteps))

        # ensure correct mask size if mask given
        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        # Loop over time steps
        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long) # tensor of timestep values matching the batch size
            
            # Sample from the diffusion model at this time step
            latent, _ = self.p_sample(latent, ts, clip_denoised=self.clip_denoised, classifier_scale=classifier_scale,
                 start_idx = 0, end_idx = -1, log_mols_every_n = log_mols_every_n)
            # Use mask if given
            if mask is not None:
                latent_orig = self.q_sample(x0, ts)
                latent = latent_orig * mask + (1. - mask) * latent

        # Return the final sample (the generated latents of molecules)
        return latent

    def q_posterior(self, x_start, x_t, t):
        """
        The mean and variance of posterior distribution q(x_{t-1}) given x_t and x_0 (start).
        """
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

        """
        Generate samples from the model.

        Args:
            batch_size (int): The number of samples to generate. Defaults to 10000.
            x_T (tensor): The final noised sample. Default is none in which case p_sample_loop will start by randomly sampling noise.
            verbose (bool): Whether to print progress information. Defaults to True.
            timesteps (int): Number of time steps to diffuse through. 
            mask (tensor): Mask for masking part of the latent. Defaults to None.
            x0 (tensor): Input tensor at time 0 (not noised).
            start_T (int): Start time step.
            sample_dim (int): The latent dimensionality of the samples. If not provided, defaults to 1024, since that is the size of the VAE latents used here.
            classifier_scale (float): A scaling factor for the classifiers. Defaults to 1.0.
            start_idx (int): Start index for diffusing and decoding in mini batches (if memory too loww for the whole batch). Defaults to 0.
            end_idx (int): End index for diffusing and decoding in mini batches. Defaults to -1 (i.e. the full batch).
            log_mols_every_n (int): Frequency for logging molecules every n time steps.
            stop_after_t (int): Time step to stop diffusion process.
            

        Returns:
            The result of the `p_sample_loop` function, which generates the samples based on the provided parameters by looping over the timesteps.
        """
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
        """
        Uses the VAE's symbol_to_idx to translate SMILES tokens into indices.

        Args:
            smiles: a SMILES string
        
        Returns:
            a tensor of indices padded with nop if necessary.
        """
        encoding = [self.symbol_to_idx[symbol] for symbol in sf.split_selfies(sf.encoder(smiles))]
        return torch.tensor(encoding + [self.symbol_to_idx['[nop]'] for i in range(self.max_len - len(encoding))])


    def smiles_to_one_hot(self, smiles):
        """
        Uses the VAE's symbol_to_idx to translate SMILES tokens into one-hot encodings.

        Args:
            smiles: a SMILES string
        
        Returns:
            a tensor of one-hot encodings representing the indeces created by smiles_to_indices(smiles)
        """
        out = torch.zeros((self.max_len, len(self.symbol_to_idx)))
        for i, index in enumerate(self.smiles_to_indices(smiles)):
            out[i][index] = 1
        return out.flatten()
        

    # used experimentally for substructure optimisation. 
    def smiles_to_z(self, smiles, vae, device):
        """
        Uses the VAE to embed SMILES into latents. 
        """
        zs = torch.zeros((len(smiles), 1024), device=device)
        for i, smile in enumerate(tqdm(smiles)):

            z = vae.encode(self.smiles_to_indices(smile).unsqueeze(0).to(device))[0].detach().requires_grad_(True)
            
            zs[i] = z.detach()
        return zs


    def one_hot_to_selfies(self, hot):
        """
        translates one-hot encodings into SELFIES.
        Args:
            hot: one-hot encoding of a SELFIES string
        Returns:
            A SELFIES string.
        """
        return ''.join([self.idx_to_symbol[idx.item()] for idx in hot.view((self.max_len, -1)).argmax(1)]).replace(' ', '')

    def one_hot_to_smiles(self, hot):
        """
        translates one-hot encodings into SMILES.
        Args:
            hot: one-hot encoding of a SELFIES string
        Returns:
            A SMILES string.
        """
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
                        # anchor_smile = "",
                        # substruct_smile = None,
                        scale_losses=True,
                        log_mols_every_n = None,
                        start_T = None,
                        stop_after_t = None):

        """
        Apply guidance to the DDPM model using a set of predictors and scales.

        Args:
            classifiers (dict): Dictionary of classifiers/predictors to use for guidance.
            properties (list): List of names that describe the properties youa re gudiing for. 
                Will be used as columns names in the output table and for reporting of progress during the sampling process.
            weights (list): List of value to weight multiple properties. Negative weights decrease the property value, positive ones increase it.
                In case of multi-property generation, the ratio between the property weights influences how strongly they are considered, e.g.
                propA has weight 1 and propB has weight 2 then propB is guided for twice as strongly.
            classifier_scales (list): List of factors to apply to classifier outputs. 
                This is the same for every property and only governs the overall guidance strength.
            sample_num (int): number of molecules to generate per classifier scale.
            exp_name (str): Name of the experiment (used for output files).
            filter (function, optional): Optional function to filter out unwanted generated molecules. Defaults to None.
            scale_losses (bool): Whether or not to use min-max-scaling on property-values during scaling. 
                Default is True to allow balanced impact on guidance for properties with very different value ranges.
            log_mols_every_n (int, optional): Interval at which to log molecule samples.
            start_T (tensor, optional): Noised latent at timepoint T wit which to start the generation process.
                Default is None, which starts by sampling random noise.
            stop_after_t (int, optional): Number of steps after which to stop sampling. Defaults to None, in which case the full number of timesteps are iterated through.

        Returns:
            None
        """
   
        pl.seed_everything(675332) # seeds for unconditional generation to caompare to limo: 215687, 156489, 
        self.load_classifiers(properties, weights, classifiers)

        # initialise the evaluator that will compute a wide variety of molecular properties for the generated molecules
        # for more in-depth evaluation after generation. The computed properties will be part of the output .csv.
        evaluator = Evaluator(self.max_len, self.symbol_to_idx)


        print(f'++++ PROPRETIES USED: {properties}')
        print(f'++++ WEIGHTS USED: {weights}')
        
        # for generation, set model to evaluation mode
        self.model.eval()
        self.sample_num = sample_num
        self.geneX = None # remnant of initial experiments with gene-expression guidance. Consider to remove.

        if scale_losses:
            print(f"++++ Using Min-max scaling for losses during sampling")
            self.scale_losses_sampling = True
        else:
            print(f"++++ Not scaling losses during sampling")
            self.scale_losses_sampling = False

        self.substruct = None

    #########################################

        # data frame that will stor all generated molecules for the different classifier scales: 
        sampled_all = pd.DataFrame()
        # emtpy tensor to append the generated molecules to when iterating through the classifier sclaes:
        mols_all = torch.Tensor().to(self.device)

        # loop over the different classifier scales that were provided and for each, apply guidance to generate molecules:
        for cs in classifier_scales:
            
            start = 0
            while self.input_dim > start:
                # data frame that will stor the generated molecules for the current classifier scale.
                sampled_cs = pd.DataFrame()

                # for memory reasons, the molecules will be smapled in batches of at most 10k
                if self.input_dim > start + 10000:
                    end = start + 10000
                else:
                    end = -1

                # use diffusion model to generate molecule latents with guidance:
                sampled_latents = self.sample(sample_dim=self.input_dim, batch_size=sample_num, classifier_scale=cs, start_idx=start, end_idx=end, 
                                              log_mols_every_n = log_mols_every_n, start_T=start_T, stop_after_t = stop_after_t)

                
                # decode latents with VAE to one-hot encodings
                if end == -1:
                    sampled_mols = torch.exp(self.vae.decode(sampled_latents[start:, :].to(self.device)))
                else:
                    sampled_mols = torch.exp(self.vae.decode(sampled_latents[start:end+1, :].to(self.device)))


                mols_all = torch.cat((mols_all, sampled_mols))

                # decode one-hot encodings to SMILES, translate into canonical SMILES and clean them (remove radicals, standardize)
                sampled_smiles = [self.one_hot_to_smiles(hot) for hot in sampled_mols]
                sampled_smiles_canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles = True) if Chem.MolFromSmiles(s) else s for s in sampled_smiles]
                sampled_smiles_canonical = [self.clean_smiles(s) for s in sampled_smiles_canonical]

                # store molecules and their scale in data frame and remove faulty ones:
                sampled_cs["smiles"] = sampled_smiles
                sampled_cs["canonical_smiles"] = sampled_smiles_canonical
                sampled_cs["guidance"] = [cs] * len(sampled_smiles)
                sampled_cs = sampled_cs[sampled_cs.canonical_smiles != 'FAULTY']

                # experimental remnant. Remove.
                if self.geneX is not None:
                    if end == -1:
                        genex_tmp = self.geneX[start:, :]
                    else:
                        genex_tmp = self.geneX[start:end, :]

                else:
                    genex_tmp = None
             
                # compute molecular properties with the evlauator:
                mol_props = evaluator.computeProperties(sampled_smiles)
                # compute predicted properties using the provided predicotrs:
                mol_props_pred = evaluator.precictProperties(sampled_smiles, sampled_mols, self.classifiers, genex = genex_tmp)
                

                for prop in self.classifiers.keys():
                        mol_props_pred[f"{prop+'_pred'}_{self.prop_weights[prop]}"] = mol_props_pred[prop+'_pred'].values
              
                mol_props = pd.concat([mol_props, mol_props_pred], axis=1)
                sampled_cs = pd.concat([sampled_cs, mol_props], axis=1)

                sampled_all = pd.concat([sampled_all, sampled_cs])

                start += 10000

            # report some property statistics of the molecules generated at the current classifier scale:
            tmp = sampled_all[sampled_all["guidance"] == cs]
            tmp = tmp.drop_duplicates("canonical_smiles")

            for prop in self.classifiers.keys():
                if prop in tmp.columns:
                    print(f"================= Min unfiltered {prop}: {min(tmp[prop].values)}")
                    print(f"====================== Median unfiltered {prop}: {median(tmp[prop].values)}")
                    print(f"====================== Mean unfiltered {prop}: {tmp[prop].mean()}")
                    print(f"=========================== Max unfiltered {prop}: {max(tmp[prop].values)}")
                else:
                    prop_ = prop+"_pred"
                    print(f"================= Min unfiltered {prop}: {min(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
                    print(f"====================== Median unfiltered {prop}: {median(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
                    print(f"=========================== Max unfiltered {prop}: {max(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
            if filter:
                # tmp = filter(tmp)
                tmp = tmp[filter(tmp.canonical_smiles)]
                for prop in self.classifiers.keys():
                    
                    print(f"================= Min filtered {prop}: {min(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
                    print(f"====================== Median filtered {prop}: {median(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
                    print(f"=========================== Max filtered {prop}: {max(tmp[f'{prop_}_{self.prop_weights[prop]}'].values)}")
           
            print(f'/////// CALCULATED DDPM CLASSIFIER {cs}///////')


        # filter applied to all mols:
        if filter:
            sampled_all = sampled_all[filter(sampled_all.canonical_smiles)]

        # save generated molecules to .csv:
        sampled_all.to_csv(exp_name + '.csv')
        # save one-hot encodings of generated molecules:
        torch.save(mols_all, exp_name + '.pt')
        

    def load_classifiers(self, prop_names, prop_weights, clf_list):
        """
        store the provided classifiers/predictors, their property-names and -weights
        """
        self.classifiers = {}
        self.prop_weights = {}

        for i, prop in enumerate(prop_names):
            self.classifiers[prop] = clf_list[i]
            self.prop_weights[prop] = prop_weights[i]

    
    @staticmethod
    def standardize(smiles):
        # Taken from: https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/

            # follows the steps in
            # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
            # as described **excellently** (by Greg) in
            # https://www.youtube.com/watch?v=eWTApNX8dJQ
            
        mol = Chem.MolFromSmiles(smiles)
        
        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol) 
        
        # if many fragments, get the "parent" (the actual mol we are interested in) 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
            
        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        
        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.
        
        te = rdMolStandardize.TautomerEnumerator() # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
        
        return taut_uncharged_parent_clean_mol

    @staticmethod
    def remove_radicals(smiles):
        # Convert SMILES to a molecule
        mol = Chem.MolFromSmiles(smiles)

        # Create a new editable molecule
        editable_mol = Chem.RWMol(mol)

        # Iterate over atoms to find radicals
        for atom in mol.GetAtoms():
            while atom.GetNumRadicalElectrons() > 0:
                # for i in range(atom.GetNumRadicalElectrons()):
    
                # Add a hydrogen atom
                hydrogen_idx = editable_mol.AddAtom(Chem.Atom(1))  # Add a hydrogen atom
                editable_mol.AddBond(atom.GetIdx(), hydrogen_idx, Chem.BondType.SINGLE)  # Create a bond to the radical atom
                # remove radikal
                atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons()-1) 


        # Convert the editable molecule back to a regular molecule
        final_mol = editable_mol.GetMol()

        # Convert back to SMILES
        # return Chem.MolToSmiles(final_mol)
        return Chem.MolToSmiles(final_mol)

    @staticmethod
    def clean_smiles(smiles, catchError = True):
        if catchError:
            try:
                smiles = LatentDiffusion.remove_radicals(smiles)
                clean_mol = LatentDiffusion.standardize(smiles)
                clean_smile = Chem.MolToSmiles(clean_mol)
                return clean_smile
            except Exception as e:
                return "FAULTY"
