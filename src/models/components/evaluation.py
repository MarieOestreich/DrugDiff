import selfies as sf
import pandas as pd
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import MolFromSmiles, MolToSmiles, QED, Descriptors
from typing import Optional, List
from tqdm import tqdm
from src.utils.sascorer import calculateScore
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import pathlib
import sys
FILE_PATH = str(pathlib.Path(__file__).parent.resolve())
FILE_PATH = "/".join(FILE_PATH.split("\\"))
print(FILE_PATH)
REPO_PATH = FILE_PATH.split('DrugDiff')[0] + 'DrugDiff'
sys.path.append(REPO_PATH)


def one_hot_to_selfies(hot, dm):
    return ''.join([dm.dataset.idx_to_symbol[idx.item()] for idx in hot.view((dm.dataset.max_len, -1)).argmax(1)]).replace(' ', '')


def one_hot_to_smiles(hot, dm):
    return sf.decoder(one_hot_to_selfies(hot, dm))

# TAKEN FROM GUACAMOL: https://github.com/BenevolentAI/guacamol

class Evaluator():
    def __init__(self, max_len=72, symbol_to_idx=None, dataset='limo'):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                


    def canonicalize(self, smiles: str, include_stereocenters=True) -> Optional[str]:
        """
        Canonicalize the SMILES strings with RDKit.
        The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543
        Args:
            smiles: SMILES string to canonicalize
            include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string
        Returns:
            Canonicalized SMILES string, None if the molecule is invalid.
        """

        mol = MolFromSmiles(smiles)

        if mol is not None:
            try:
                m = MolToSmiles(mol, isomericSmiles=include_stereocenters)
                return m
            except:
                return None
        else:
            return None
        
    def canonicalize_smiles(self, smiles_list, include_stereocenters=True) -> List[str]:
        """
        Canonicalize a list of smiles. Filters out repetitions and removes corrupted molecules.
        Args:
            smiles_list: molecules as SMILES strings
            include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings
        Returns:
            The canonicalized and filtered input smiles.
        """

        canonicalized_smiles = [self.canonicalize(smiles, include_stereocenters) for smiles in smiles_list]

        # Remove None elements
        canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]
     
        unique_list= self.remove_duplicates_smiles(canonicalized_smiles)
        return unique_list
    
    def canonicalize_list(self, smiles_list, mols_list, include_stereocenters=True) -> List[str]:
        """
        Canonicalize a list of smiles. Filters out repetitions and removes corrupted molecules.
        Args:
            smiles_list: molecules as SMILES strings
            include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings
        Returns:
            The canonicalized and filtered input smiles.
        """

        canonicalized_smiles = [self.canonicalize(smiles, include_stereocenters) for smiles in smiles_list]

        # Remove None elements
        canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]
        canonicalized_mols_ind = [ind for ind, s in enumerate(canonicalized_smiles) if s is not None]
        canonicalized_mols = mols_list[canonicalized_mols_ind]
        unique_list, mols = self.remove_duplicates(canonicalized_smiles, canonicalized_mols)
        return unique_list, mols

    def remove_duplicates_smiles(self, list_with_duplicates):
        """
        Removes the duplicates and keeps the ordering of the original list.
        For duplicates, the first occurrence is kept and the later occurrences are ignored.
        Args:
            list_with_duplicates: list that possibly contains duplicates
        Returns:
            A list with no duplicates.
        """

        unique_set = set()
        unique_list = []
        indices = []
        for ind, element in enumerate(list_with_duplicates):
            if element not in unique_set:
                unique_set.add(element)
                unique_list.append(element)
                indices.append(ind)
        return unique_list    
    
    def remove_duplicates(self, list_with_duplicates, mols):
        """
        Removes the duplicates and keeps the ordering of the original list.
        For duplicates, the first occurrence is kept and the later occurrences are ignored.
        Args:
            list_with_duplicates: list that possibly contains duplicates
        Returns:
            A list with no duplicates.
        """

        unique_set = set()
        unique_list = []
        indices = []
        for ind, element in enumerate(list_with_duplicates):
            if element not in unique_set:
                unique_set.add(element)
                unique_list.append(element)
                indices.append(ind)
        mols = mols[indices]        
        return unique_list, mols
########################

    def smiles_to_logp(self, smiles):
        logps = []
        for i, s in enumerate(tqdm(smiles, desc='calculating logP')):
            smile = s
            try:
                logps.append(MolLogP(MolFromSmiles(smile)))
            except:
                logps.append(0)
        return logps

    def smiles_to_plogp(self, smiles):
        logps = []
        for i, s in enumerate(tqdm(smiles, desc='calculating p-logP')):
            mol = MolFromSmiles(s)
            penalized_logp = MolLogP(mol) - calculateScore(mol)
            for ring in mol.GetRingInfo().AtomRings():
                if len(ring) > 6:
                    penalized_logp -= 1
            logps.append(penalized_logp)
        return logps

    def smiles_to_qed(self, smiles):
        qeds = []
        for i, smile in enumerate(tqdm(smiles, desc='calculating QED')):
            mol = MolFromSmiles(smile)
            if mol is not None:
                qeds.append(QED.qed(mol))
            else:
                qeds.append(0.)
        return qeds

    def smiles_to_cycles(self, smiles):
        cycles = []
        for smile in tqdm(smiles, desc='counting undesired cycles'):
            mol = MolFromSmiles(smile)
            if mol is not None:
                cycle_count = 0
                for ring in mol.GetRingInfo().AtomRings():
                    if not (4 < len(ring) < 7):
                        cycle_count += 1
                cycles.append(cycle_count)
            else:
                cycles.append(1881)
        return cycles


    def tanimoto_calc(self, smi1, smi2):
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
        s = round(DataStructs.TanimotoSimilarity(fp1,fp2),3)
        return s


    def smiles_to_tanimoto(self, smiles1, smiles2):
        sims = []
        for s1, s2 in zip(smiles1, smiles2):
            sims.append(self.tanimoto_calc(s1, s2))
        return sims

    def smiles_to_sa(self, smiles):
        sas = []
        for i, s in enumerate(tqdm(smiles, desc='calculating SA')):
            smile = s
            mol = MolFromSmiles(smile)
            if mol is not None:
                sas.append(calculateScore(mol))
            else:
                sas.append(100)     
        return sas

    
    def mols_to_pred(self, mols, predictor, device):
        predictor.eval()
        predictor = predictor.to(device)
        preds = predictor(mols.to(device))
   
        return preds.squeeze().cpu().detach().numpy()

    

    def smiles_to_molweight(self, smiles):
        weigths = []
        for s in smiles:
            w = Chem.Descriptors.ExactMolWt(MolFromSmiles(s))
            weigths.append(w)
        return weigths

    def check_substruct(self, smiles, substruct):
        sub=[]
        for s in smiles:
            m = MolFromSmiles(s)
            if m.HasSubstructMatch(substruct):
                sub.append(1)
            else:
                sub.append(0)
        return sub



    # def computeProperties(self, smiles, mols, props , genes=None, substruct_smile=None, custom_preds={}):
    def computeProperties(self, smiles):
   
        mol_props = pd.DataFrame()
        mol_props['cycles'] = self.smiles_to_cycles(smiles)
        mol_props['sa'] = self.smiles_to_sa(smiles)
        mol_props['qed'] = self.smiles_to_qed(smiles)
        mol_props['logp'] = self.smiles_to_logp(smiles)
        mol_props['plogp'] = self.smiles_to_plogp(smiles)
        mol_props['mol_weight'] = self.smiles_to_molweight(smiles)
        mol_props['h_bond_donors'] = [Chem.Descriptors.NumHDonors(MolFromSmiles(s)) for s in smiles]
        mol_props['h_bond_acceptors'] = [Chem.Descriptors.NumHAcceptors(MolFromSmiles(s)) for s in smiles]
        mol_props['rotatable_bonds'] = [Chem.Descriptors.NumRotatableBonds(MolFromSmiles(s)) for s in smiles]
        mol_props['number_of_atoms'] = [Chem.rdchem.Mol.GetNumAtoms(MolFromSmiles(s)) for s in smiles]
        mol_props['molar_refractivity'] = [Chem.Crippen.MolMR(MolFromSmiles(s)) for s in smiles]
        mol_props['topological_surface_area_mapping'] = [Chem.QED.properties(MolFromSmiles(s)).PSA for s in smiles]
        mol_props['formal_charge'] = [Chem.rdmolops.GetFormalCharge(MolFromSmiles(s)) for s in smiles]
        mol_props['heavy_atoms'] = [Chem.rdchem.Mol.GetNumHeavyAtoms(MolFromSmiles(s)) for s in smiles]
        mol_props['num_of_rings'] = [Chem.rdMolDescriptors.CalcNumRings(MolFromSmiles(s)) for s in smiles]
        return mol_props
    
    def precictProperties(self, smiles, mols, custom_preds={}, genex = None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        mol_props = pd.DataFrame()
        for prop in custom_preds.keys():
            mol_props[prop+'_pred'] = self.mols_to_pred(mols, custom_preds[prop], device)
        return mol_props



    
