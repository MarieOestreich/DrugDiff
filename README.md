# DrugDiff

_DrugDiff_ is a latent diffusion model that uses predictor guidance to generate small molecules with desired molecular properties.
Details about DrugDiff's architecture are illustrated below and further described in our [pre-print](https://doi.org/10.1101/2024.07.17.603873).

![Figure 1](https://github.com/MarieOestreich/DrugDiff/blob/main/DrugDiff-Overview.jpg)
__Figure 1: DrugDiff Oveview__

## 1. Download Model Checkpoints

Please download the DrugDiff checkpoint as well as the predictor checkpoints from here: [zenodo](https://zenodo.org/records/12755763)

Save the checkpoints under ```/model/```. 

## 2. Generate Molecules

### Unconditional Generation
To generate molecules from the learned distribution without guidance, use ```/scripts/generate_without_guidance.py```.

### Single-Property Guidance
To generate molecules with guidance towards a single molecular property, use ```/scripts/generate_with_single_property_guidance.py```.

### Multi-Property Guidance
To generate molecules with guidance towards multiple molecular properties, use ```/scripts/generate_with_multi_property_guidance.py```.

## 3. Evaluate Generated Molecules
Evaluation examples can be found under ```/notebooks/```, with examples for unconditional generation (```evaluation_without_guidance.ipynb```), single-property guidance (```evaluation_with_single_property_guidance.ipynb```) and multi-property guidance (```evaluation_with_multi_property_guidance.ipynb```).

## Data
The ZINC-250k dataset (ht</span>tps://</span>w</span>ww.kaggle.com/datasets/basu369victor/zinc250k) was used for training, which was published under the Database Contents License (DbCL). It is a subset of the ZINC database (Irwin, Tang, Young, Dandarchuluun, Wong, Khurelbaatar, Moroz, Mayfield, Sayle, J. Chem. Inf. Model 2020, ht</span>tps\://</span>pubs.acs.org/doi/10.1021/acs.jcim.0c00675).
