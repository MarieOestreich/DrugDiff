# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


class DenseFiLM(nn.Module):
  """Feature-wise linear modulation (FiLM) generator."""
  def __init__(self, embedding_channels=128, output_channels=2048):
    super().__init__()
    self.embedding_channels = embedding_channels
    self.lin1 = nn.Linear(embedding_channels, embedding_channels*4)
    self.lin2 = nn.Linear(embedding_channels*4, embedding_channels*4)
    self.lin3 = nn.Linear(embedding_channels*4, output_channels)
    self.lin4 = nn.Linear(embedding_channels*4, output_channels)
    self.act = nn.SiLU()
  def forward(self, position):
    pos_encoding = get_timestep_embedding(position, self.embedding_channels)
    pos_encoding = self.lin1(pos_encoding)
    pos_encoding = self.act(pos_encoding)
    pos_encoding = self.lin2(pos_encoding)
    scale = self.lin3(pos_encoding)
    shift = self.lin4(pos_encoding)
    return scale, shift

class CondDenseFiLM(nn.Module):
  """Feature-wise linear modulation (FiLM) generator."""
  def __init__(self, embedding_channels=128, output_channels=2048):
    super().__init__()
    self.embedding_channels = embedding_channels
    self.lin1 = nn.Linear(embedding_channels, embedding_channels*4)
    self.lin2 = nn.Linear(embedding_channels*4, embedding_channels*4)
    self.lin3 = nn.Linear(embedding_channels*4, output_channels)
    self.lin4 = nn.Linear(embedding_channels*4, output_channels)
    self.act = nn.SiLU()
  def forward(self, cond_vec):
    cond_vec = self.lin1(cond_vec)
    cond_vec = self.act(cond_vec)
    cond_vec = self.lin2(cond_vec)
    scale = self.lin3(cond_vec)
    shift = self.lin4(cond_vec)
    return scale, shift

class DenseResBlock(nn.Module):
  """Fully-connected residual block."""
  def __init__(self, input_dim=2048, output_size=2048):
    super().__init__()
    self.ln = torch.nn.LayerNorm(input_dim)
    self.lin1= torch.nn.Linear(input_dim, output_size)
    self.lin2= torch.nn.Linear(output_size, output_size)
    self.res_lin = nn.Linear(input_dim, output_size) if input_dim != output_size else nn.Identity()
    self.act = nn.SiLU()
  def forward(self, inputs, scale, shift):
    output = self.ln(inputs)
    output = scale * output + shift
    output = self.act(output)
    output = self.lin1(output)
    output = self.ln(output)
    output = scale * output + shift
    output = self.act(output)
    output = self.lin2(output)
    shortcut = inputs
    shortcut = self.res_lin(shortcut)
    return output + shortcut

class DenseResBlockCat(nn.Module):
  """Fully-connected residual block."""
  def __init__(self, input_dim=2048, output_size=2048):
    super().__init__()
    self.ln = torch.nn.LayerNorm(input_dim)
    self.lin1= torch.nn.Linear(input_dim, input_dim)
    self.lin2= torch.nn.Linear(input_dim, output_size)
    self.res_lin = nn.Linear(input_dim, output_size) if input_dim != output_size else nn.Identity()
    self.act = nn.SiLU()
  def forward(self, inputs, scale, shift):
    output = self.ln(inputs)
    output = scale * output + shift
    output = self.act(output)
    output = self.lin1(output)
    output = self.ln(output)
    output = scale * output + shift
    output = self.act(output)
    output = self.lin2(output)
    shortcut = inputs
    shortcut = self.res_lin(shortcut)
    return output + shortcut


class DenseDDPM(nn.Module):
    """Fully-connected diffusion network."""
    def __init__(self, input_dim=1278, mlp_dims=2048): # input - output = concat condition (300 mols)
        super().__init__()
        


        self.lin1= nn.Linear(input_dim, mlp_dims)
        self.lin2= nn.Linear(mlp_dims,input_dim)

        self.ln = nn.LayerNorm(mlp_dims)

        self.dense_film1 = DenseFiLM(128, mlp_dims)
        self.dense_film2 = DenseFiLM(128, mlp_dims)
        self.dense_film3 = DenseFiLM(128, mlp_dims)
        self.dense_res1 = DenseResBlock(mlp_dims)
        self.dense_res2 = DenseResBlock(mlp_dims)
        self.dense_res3 = DenseResBlock(mlp_dims)


    def forward(self, x, t):


        x = self.lin1(x)

        scale, shift = self.dense_film1(t)
        x = self.dense_res1(x, scale=scale, shift=shift)


        scale, shift = self.dense_film2(t)
        x = self.dense_res2(x, scale=scale, shift=shift)


        scale, shift = self.dense_film3(t)
        x = self.dense_res3(x,scale=scale, shift=shift)

        x = self.ln(x)
        x = self.lin2(x)
        return x

class DenseDDPMCat(nn.Module):
    """Fully-connected diffusion network."""
    def __init__(self, input_dim=1278, output_dim=978, mlp_dims=2048, cond_input_dim = 1): # input - output = concat condition (300 mols)
        super().__init__()
        self.mlp_dims = mlp_dims
        
        self.pretrained_mlp = False
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, mlp_dims),
            nn.SiLU(),
            nn.Linear(mlp_dims, mlp_dims)
        )

        self.lin1= nn.Linear(input_dim, mlp_dims)
        self.lin2= nn.Linear(mlp_dims,output_dim)

        self.ln = nn.LayerNorm(mlp_dims)

        self.dense_film1 = DenseFiLM(128, 2 * mlp_dims)
        self.dense_film2 = DenseFiLM(128, 2 * mlp_dims)
        self.dense_film3 = DenseFiLM(128, 2 * mlp_dims)
        self.dense_res1 = DenseResBlockCat(2 * mlp_dims, mlp_dims)
        self.dense_res2 = DenseResBlockCat(2 * mlp_dims, mlp_dims)
        self.dense_res3 = DenseResBlockCat(2 * mlp_dims, mlp_dims)


    def forward(self, x, cond_num, cond_switch, t, p = 0.9):
        
        # print(f'shape condition: {cond_num.size()}')

        # randomly switch off conditioning with probability 10%
        if cond_switch == 1:
            if np.random.uniform(0,1) > p:
                cond_switch == 0

        if cond_switch == 1:
            cond = self.cond_mlp(cond_num)
        else:
            # cond = self.cond_mlp(cond_num)
            cond = torch.zeros((x.size()[0], self.mlp_dims)).to('cuda')


        x = self.lin1(x) 
        
        x = torch.cat([cond, x], dim = 1)

        scale, shift = self.dense_film1(t)
        x = self.dense_res1(x, scale=scale, shift=shift) 
        x = torch.cat([cond, x], dim = 1) 
          

        scale, shift = self.dense_film2(t)
        x = self.dense_res2(x, scale=scale, shift=shift)
        x = torch.cat([cond, x], dim = 1) 

        scale, shift = self.dense_film3(t)
        x = self.dense_res3(x,scale=scale, shift=shift) 


        x = self.ln(x)
        x = self.lin2(x)
        return x

    
    
class DenseDDPMMultiCond(nn.Module):
    """Fully-connected diffusion network."""
    def __init__(self, input_dim=1278, output_dim=978, mlp_dims=2048, cond_input_dim = []): # input - output = concat condition (300 mols)
        super().__init__()
        self.mlp_dims = mlp_dims
        
        self.pretrained_mlp = False
        self.cond_mlp_list = nn.ModuleList()
        
        for i in cond_input_dim:
            cond_mlp = nn.Sequential(
                nn.Linear(i, mlp_dims),
                nn.SiLU(),
                nn.Linear(mlp_dims, mlp_dims)
            )
            self.cond_mlp_list.append(cond_mlp)
        

        self.lin1= nn.Linear(input_dim, mlp_dims)
        self.lin2= nn.Linear(mlp_dims,output_dim)

        self.ln = nn.LayerNorm(mlp_dims)

        self.dense_film1 = DenseFiLM(128, (len(cond_input_dim)+1) * mlp_dims)
        self.dense_film2 = DenseFiLM(128, (len(cond_input_dim)+1) * mlp_dims)
        self.dense_film3 = DenseFiLM(128, (len(cond_input_dim)+1) * mlp_dims)
        self.dense_res1 = DenseResBlockCat((len(cond_input_dim)+1) * mlp_dims, mlp_dims)
        self.dense_res2 = DenseResBlockCat((len(cond_input_dim)+1) * mlp_dims, mlp_dims)
        self.dense_res3 = DenseResBlockCat((len(cond_input_dim)+1) * mlp_dims, mlp_dims)


    def forward(self, x, cond_num, cond_switch, t, p = 0.9):
        
        # print(f'shape condition: {cond_num.size()}')

        # randomly switch off conditioning with probability 10%
        if cond_switch == 1:
            if np.random.uniform(0,1) > p:
                cond_switch == 0

        if cond_switch == 1:
            conds = []
            for i in range(len(cond_num)):
                conds.append(self.cond_mlp_list[i](cond_num[i]))
            cond = torch.cat(conds, dim=1)
        else:
            # cond = self.cond_mlp(cond_num)
            cond = torch.zeros((x.size()[0], self.mlp_dims*len(self.cond_mlp_list))).to('cuda')


        x = self.lin1(x) 
        
        x = torch.cat([cond, x], dim = 1) 

        scale, shift = self.dense_film1(t)
        x = self.dense_res1(x, scale=scale, shift=shift) 
        x = torch.cat([cond, x], dim = 1) 
          

        scale, shift = self.dense_film2(t)
        x = self.dense_res2(x, scale=scale, shift=shift) 
        x = torch.cat([cond, x], dim = 1) 

        scale, shift = self.dense_film3(t)
        x = self.dense_res3(x,scale=scale, shift=shift) 


        x = self.ln(x)
        x = self.lin2(x)
        return x

class DenseDDPMConditional(nn.Module):
    """Fully-connected diffusion network."""
    def __init__(self, input_dim=1024, output_dim=1024, mlp_dims=2048, initial_cond_dim = 128, cond='logp', mode='unconditional', gaussian_kernels=50): # input - output = concat condition (300 mols)
        super().__init__()
        
        self.cond_layer_list = nn.ModuleList()
        self.cond = cond
        if cond == 'scaffold':
            self.cond_mlp  = FingerprintEmbedding(2048,initial_cond_dim)
        elif 'gene_raw' in cond:
            self.cond_mlp  = GeneExprEmbedding(978,initial_cond_dim)
        elif 'gene_emb' in cond:
            self.cond_mlp  = GeneExprEmbedding(1024,initial_cond_dim)
        elif cond == 'cline':
            n_in=100
            self.cond_mlp  = ClassEmbedding(n_classes=10, n_in=n_in, n_out=initial_cond_dim)
        elif cond == 'mol_emb':
            n_in=1024
            self.cond_mlp  = MoleculeEmbedding(n_in=n_in, n_out=initial_cond_dim)
        elif cond in ["LogP", 'SA', 'QED', 'esr1_ba', 'acaa1_ba']:
            if cond=='LogP':
                start = -10.
                stop = 10.
            elif cond=='SA':
                start = 1.
                stop = 7.
            elif cond=='QED':
                start = 0.1
                stop = 1.
            else:
                start = -1.
                stop = -8.
            n_in=gaussian_kernels
            self.cond_mlp = PropertyEmbedding(n_in=n_in, n_out=initial_cond_dim, start=start, stop=stop)

        self.mode = mode
        self.lin1= nn.Linear(input_dim, mlp_dims)
        self.lin2= nn.Linear(mlp_dims,output_dim)
        self.ln = nn.LayerNorm(mlp_dims)

        if self.mode == 'scale_shift':
            
            self.cond_dense_film1 = CondDenseFiLM(initial_cond_dim, mlp_dims)
            self.cond_dense_film2 = CondDenseFiLM(initial_cond_dim, mlp_dims)
            self.cond_dense_film3 = CondDenseFiLM(initial_cond_dim, mlp_dims)
            self.cond_dense_res1 = DenseResBlock(mlp_dims, mlp_dims)
            self.cond_dense_res2 = DenseResBlock(mlp_dims, mlp_dims)
            self.cond_dense_res3 = DenseResBlock(mlp_dims, mlp_dims)


        self.dense_film1 = DenseFiLM(128, mlp_dims)
        self.dense_film2 = DenseFiLM(128, mlp_dims)
        self.dense_film3 = DenseFiLM(128, mlp_dims)
        self.dense_res1 = DenseResBlock(mlp_dims, mlp_dims)
        self.dense_res2 = DenseResBlock(mlp_dims, mlp_dims)
        self.dense_res3 = DenseResBlock(mlp_dims, mlp_dims)


    def forward(self, x, cond_inputs, cond_mask, t):


        if self.cond in ["LogP", 'SA', 'QED', 'esr1_ba', 'acaa1_ba']:
            
            final_cond_vec = self.cond_mlp(cond_inputs).squeeze()
        else:
            final_cond_vec = self.cond_mlp(cond_inputs.float())
      
        x = self.lin1(x)
        if self.mode == 'scale_shift' and cond_mask>0.1:
            scale, shift = self.cond_dense_film1(final_cond_vec)
            x = self.cond_dense_res1(x, scale=scale, shift=shift)

        scale, shift = self.dense_film1(t)
        x = self.dense_res1(x, scale=scale, shift=shift)
        
        if self.mode == 'scale_shift' and cond_mask>0.1:
            scale, shift = self.cond_dense_film2(final_cond_vec)
 
            x = self.cond_dense_res2(x, scale=scale, shift=shift)

        scale, shift = self.dense_film2(t)
        x = self.dense_res2(x, scale=scale, shift=shift)
        
        if self.mode == 'scale_shift' and cond_mask>0.1:
            scale, shift = self.cond_dense_film3(final_cond_vec)
            x = self.cond_dense_res3(x, scale=scale, shift=shift)

        scale, shift = self.dense_film3(t)
        x = self.dense_res3(x,scale=scale, shift=shift)

        x = self.ln(x)
        x = self.lin2(x)
        return x

class DenseDDPMUnConditional(nn.Module):
    """Fully-connected diffusion network."""
    def __init__(self, input_dim=1024, output_dim=1024, mlp_dims=2048, initial_cond_dim = 128, cond='logp', mode='unconditional', gaussian_kernels=50): # input - output = concat condition (300 mols)
        super().__init__()
        
        self.cond_layer_list = nn.ModuleList()
        self.cond = cond
        if cond == 'scaffold':
            self.cond_mlp  = FingerprintEmbedding(2048,initial_cond_dim)
        elif 'gene_raw' in cond:
            self.cond_mlp  = GeneExprEmbedding(978,initial_cond_dim)
        elif 'gene_emb' in cond:
            self.cond_mlp  = GeneExprEmbedding(1024,initial_cond_dim)
        elif cond == 'cline':
            n_in=100
            self.cond_mlp  = ClassEmbedding(n_classes=10, n_in=n_in, n_out=initial_cond_dim)
        elif cond == 'mol_emb':
            n_in=1024
            self.cond_mlp  = MoleculeEmbedding(n_in=n_in, n_out=initial_cond_dim)
        elif cond in ["LogP", 'SA', 'QED', 'esr1_ba', 'acaa1_ba']:
            if cond=='LogP':
                start = -10.
                stop = 10.
            elif cond=='SA':
                start = 1.
                stop = 7.
            elif cond=='QED':
                start = 0.1
                stop = 1.
            else:
                start = -1.
                stop = -8.
            n_in=gaussian_kernels
            self.cond_mlp = PropertyEmbedding(n_in=n_in, n_out=initial_cond_dim, start=start, stop=stop)

        self.mode = mode
        self.lin1= nn.Linear(input_dim, mlp_dims)
        self.lin2= nn.Linear(mlp_dims,output_dim)
        self.ln = nn.LayerNorm(mlp_dims)

        if self.mode == 'scale_shift':
            
            self.cond_dense_film1 = CondDenseFiLM(initial_cond_dim, mlp_dims)
            self.cond_dense_film2 = CondDenseFiLM(initial_cond_dim, mlp_dims)
            self.cond_dense_film3 = CondDenseFiLM(initial_cond_dim, mlp_dims)
            self.cond_dense_res1 = DenseResBlock(mlp_dims, mlp_dims)
            self.cond_dense_res2 = DenseResBlock(mlp_dims, mlp_dims)
            self.cond_dense_res3 = DenseResBlock(mlp_dims, mlp_dims)


        self.dense_film1 = DenseFiLM(128, mlp_dims)
        self.dense_film2 = DenseFiLM(128, mlp_dims)
        self.dense_film3 = DenseFiLM(128, mlp_dims)
        self.dense_res1 = DenseResBlock(mlp_dims, mlp_dims)
        self.dense_res2 = DenseResBlock(mlp_dims, mlp_dims)
        self.dense_res3 = DenseResBlock(mlp_dims, mlp_dims)


    def forward(self, x, cond_inputs, cond_mask, t):


        if self.cond in ["LogP", 'SA', 'QED', 'esr1_ba', 'acaa1_ba']:
            
            final_cond_vec = self.cond_mlp(cond_inputs).squeeze()
        else:
            final_cond_vec = self.cond_mlp(cond_inputs.float())
      
        x = self.lin1(x)
        if self.mode == 'scale_shift' and cond_mask>0.1:
            scale, shift = self.cond_dense_film1(final_cond_vec)
            x = self.cond_dense_res1(x, scale=scale, shift=shift)

        scale, shift = self.dense_film1(t)
        x = self.dense_res1(x, scale=scale, shift=shift)
        
        if self.mode == 'scale_shift' and cond_mask>0.1:
            scale, shift = self.cond_dense_film2(final_cond_vec)
 
            x = self.cond_dense_res2(x, scale=scale, shift=shift)

        scale, shift = self.dense_film2(t)
        x = self.dense_res2(x, scale=scale, shift=shift)
        
        if self.mode == 'scale_shift' and cond_mask>0.1:
            scale, shift = self.cond_dense_film3(final_cond_vec)
            x = self.cond_dense_res3(x, scale=scale, shift=shift)

        scale, shift = self.dense_film3(t)
        x = self.dense_res3(x,scale=scale, shift=shift)

        x = self.ln(x)
        x = self.lin2(x)
        return x