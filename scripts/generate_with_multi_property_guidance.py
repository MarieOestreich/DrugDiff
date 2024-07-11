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
from src.models.components.vae import PropertyPredictor

################################################################################################
# set model parameters

input_dim = 1024 # do not change
mlp_dims = 2048 # do not change
num_sample_mols = 10000 # how many molecules to generate
uncond_dm_ckpt = REPO_PATH + '/model/drugdiff.ckpt'
output_file = REPO_PATH+'/outputs/num_atoms_and_logp_generation'

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
# load classifiers (here as an example: number of atoms and logp)

classifier_num_atoms = PropertyPredictor.load_from_checkpoint(REPO_PATH + '/model/number_of_atoms.ckpt', in_dim=model.max_len*len(model.symbol_to_idx))
classifier_num_atoms.eval()
classifier_num_atoms = classifier_num_atoms.to(device)

classifier_logp = PropertyPredictor.load_from_checkpoint(REPO_PATH + '/model/logp.ckpt', in_dim=model.max_len*len(model.symbol_to_idx))
classifier_logp.eval()
classifier_logp = classifier_logp.to(device)

################################################################################################
# generate

model.apply_guidance(classifiers = [classifier_num_atoms, classifier_logp], 
                    # negative weights decrease the property value, positive ones increase it
                    # in case of multi-property generation, the ratio between the property weights
                    # influences how strongly they are considered, e.g.
                    # propA has weight 1 and propB has weight 2 than propB is guided for twice as strongly.
                    weights = [-1, -1], 
                    properties = ['num_atoms', 'logp'],
                    classifier_scales = [0, 100], # we include 0 (unguided) to have a reference for evaluation
                    sample_num = num_sample_mols, # how many molecules to generate (as defined above)
                    exp_name = output_file)