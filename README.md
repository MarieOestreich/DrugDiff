# DrugDiff

## 1. Download Model Checkpoints

Please download the DrugDiff checkpoint as well as the predictor checkpoints from here: ___
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