import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from src.utils import remove_isotopes_from_smiles
from src.utils import remove_isotopes_from_mol


def top10(y_true, y_pred):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=10)


def top50(y_true, y_pred):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=50)


class RpFingerprintModel():
    """A template Prioritizer based on template relevance.

    Attributes:
        fp_length (int): Fingerprint length.
        fp_radius (int): Fingerprint radius.
    """

    def __init__(self,
                 rp_model_path: str = "./",
                 variance_path: str = "./",
                 fp_length: int = 65536,
                 fp_radius: int = 2):
        # run on cpu
        tf.config.set_visible_devices([], 'GPU')

        self.rp_model = tf.keras.models.load_model(rp_model_path, custom_objects={'top10': top10, 'top50': top50})
        self.variance = np.load(variance_path)
        self.variance_mask = np.argpartition(self.variance, -16384)[-16384:]
        self.fp_length = fp_length
        self.fp_radius = fp_radius

    def smiles_to_fp(self, smiles: str):
        """Converts SMILES string to fingerprint for use with template relevance model.

        Args:
            smiles (str): SMILES string to convert to fingerprint

        Returns:
            np.ndarray of np.float32: Fingerprint for given SMILES string.

        """
        clean_smiles = remove_isotopes_from_smiles(smiles)
        mol = Chem.MolFromSmiles(clean_smiles)
        if not mol:
            return np.zeros((self.fp_length,), dtype=np.float32)
        return np.array(
            AllChem.GetMorganFingerprintAsBitVect(
                mol, self.fp_radius, nBits=self.fp_length, useChirality=True
            ), dtype=np.float32)

    def predict(self, smiles: list):
        fp = []
        for smi in smiles:
            fp_tmp = self.smiles_to_fp(smi)
            fp_tmp = fp_tmp[self.variance_mask]
            fp.append(fp_tmp)
        fp = np.array(fp)
        scores = self.rp_model.predict(fp, batch_size=len(smiles))#.reshape(-1)
        return scores


class FfFingerprintModel():
    """A template Prioritizer based on template relevance.

    Attributes:
        fp_length (int): Fingerprint length.
        fp_radius (int): Fingerprint radius.
    """

    def __init__(self,
                 ff_model_path: str = "./",
                 fp_length: int = 8192):

        if ff_model_path.split(".")[-1] == "h5":
            self.backend = "tf"
        elif (ff_model_path.split(".")[-1] == "pt") or (ff_model_path.split(".")[-1] == "pth"):
            self.backend = "torch"
        else:
            print("Fast filter model backend not recognized")
            return None

        self.fp_length = fp_length
        
        if self.backend == "tf":
            # run on cpu
            tf.config.set_visible_devices([], 'GPU')
            self.ff_model = tf.keras.models.load_model(ff_model_path)
        if self.backend == "torch":
            # run on cpu
            device = torch.device("cpu")
            self.ff_model = torch.jit.load(ff_model_path)
            self.ff_model.to(device)
            self.ff_model.eval()
            self.logit_fn = nn.Sigmoid()
            self.fp_gen_ecfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=self.fp_length)
            
    def gen_fp(self, reactant_smiles: str, target: str):
        """Evaluates likelihood of given reaction (Bayer version).

        Args:
            reactant_smiles (str): SMILES string of reactants.
            target (str): SMILES string of target product.
            **kwargs: Unused.

        Returns:
            A list of reaction outcomes.
        """
        targetMol = Chem.MolFromSmiles(target)
        reactantMol = Chem.MolFromSmiles(reactant_smiles)
        if (targetMol is None) or (reactantMol is None):
            return None
        targetMol = remove_isotopes_from_mol(targetMol)
        reactantMol = remove_isotopes_from_mol(reactantMol)

        fp_prd_arr = np.zeros(1, dtype=int)
        fp_rct_arr = np.zeros(1, dtype=int)
        if self.backend == "tf":
            fp_prd = AllChem.GetMorganFingerprintAsBitVect(targetMol, 2, nBits=self.fp_length)
            fp_rct = AllChem.GetMorganFingerprintAsBitVect(reactantMol, 2, nBits=self.fp_length)
        if self.backend == "torch":
            fp_prd = self.fp_gen_ecfp.GetFingerprint(targetMol)
            fp_rct = self.fp_gen_ecfp.GetFingerprint(reactantMol)
        DataStructs.ConvertToNumpyArray(fp_prd, fp_prd_arr)
        DataStructs.ConvertToNumpyArray(fp_rct, fp_rct_arr)
        rxnfp = np.concatenate((fp_rct_arr, fp_prd_arr), axis=None)
        return rxnfp

    def predict(self, fingerprints: np.ndarray):
        if self.backend == "tf":
            scores = self.ff_model.predict(fingerprints)
        if self.backend == "torch":
            logits = self.ff_model(torch.from_numpy(fingerprints).to(torch.float32)).detach()
            scores = self.logit_fn(logits)
        return scores
