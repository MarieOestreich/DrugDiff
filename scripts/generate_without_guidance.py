import torch
import sys
import pathlib
from pathlib import Path

FILE_PATH = str(Path(__file__).parent.resolve())
FILE_PATH = "/".join(FILE_PATH.split("\\"))
print(FILE_PATH)
REPO_PATH = FILE_PATH.split('DrugDiff')[0] + 'DrugDiff'
sys.path.append(REPO_PATH)
Path(REPO_PATH+'/outputs').mkdir(parents=True, exist_ok=True)

from src.models.ddpm import LatentDiffusion

################################################################################################
# set model parameters

input_dim = 1024 # do not change
mlp_dims = 2048 # do not change
num_sample_mols = 10000 # set, how many molecules to generate
uncond_dm_ckpt = REPO_PATH + '/model/drugdiff.ckpt'
# output_file = REPO_PATH+'/outputs/unconditional_generation'
output_file = REPO_PATH+'/outputs/unconditional_comparison_limo_drugdiff/10k_drugdiff_2.csv'


################################################################################################
# load DrugDiff

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model = LatentDiffusion(input_dim = input_dim,
                        output_dim = input_dim,
                        mlp_dims = mlp_dims,
                        num_sample_mols = num_sample_mols,
                        vae_ckpt = None,
                        uncond_dm_ckpt = uncond_dm_ckpt,
                        dataset = 'zinc250k')

model.to(device)
model.eval()

################################################################################################
# generate

model.apply_guidance(classifiers = [], # empty for unconditional generation
                    weights = [1],
                    properties = [],
                    classifier_scales = [0], # 0 stands for no guidance
                    sample_num = num_sample_mols, # how many molecules to generate (as defined above)
                    exp_name = output_file)