import gzip
import json
import pickle
import sys
import rdkit.Chem as Chem


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print('Boolean value expected.')
        sys.exit()


def load_data(template_path: str, buyables_path: str):
    with gzip.open(template_path, 'rb') as f:
        templates = json.loads(f.read().decode('utf-8'))
    
    if ".pkl" in buyables_path:
        with open(buyables_path, 'rb') as f:
            buyables = pickle.load(f)
    elif ".json" in buyables_path:
        with gzip.open(buyables_path, 'rt', encoding='utf-8') as f:
            buyables_list = json.load(f)
        buyables = {}
        for bb in buyables_list:
            buyables[bb["smiles"]] = bb["ppg"]
    else:
        print("unknown file type of buyables file {}".format(buyables_path))

    return templates, buyables


def remove_isotopes_from_smiles(smiles: str):
    """ Removes the isotope label of each atom of a SMILES.
    If the provided smiles string is not canonical it will be after
    the removal.

    Parameters
    ----------
    smi : str
        A SMILES string.
    Returns
    -------
    str
        A SMILES string without isotopes.
    """
    mol_tmp = Chem.MolFromSmiles(smiles)
    atom_data = [(atom, atom.GetIsotope()) for atom in mol_tmp.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            atom.SetIsotope(0)
    clean_smiles = Chem.MolToSmiles(mol_tmp, True)
    return clean_smiles


def remove_isotopes_from_mol(mol: object):
    """ Removes the isotope label of each atom of a RDKit targets.

    Parameters
    ----------
    mol : object
        A RDKit molecule object.
    Returns
    -------
    str
        A RDKit molecule object without isotopes.
    """
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            atom.SetIsotope(0)
    return mol


# The following code was adapted from "ASKCOS: Software Tools for Organic Synthesis, MLPDS, 2020, https://github.com/ASKCOS/ASKCOS"
class Molecule(object):
    def __init__(self, node, depth=0):
        
        self.smiles = getCanonicalSmiles(node['smiles'])
        self.price = node['ppg']
        self.as_reactant = node['as_reactant']
        self.as_product = node['as_product']
        self.depth = depth
        self.is_terminal = len(node['children']) == 0
        
        if self.is_terminal:
            self.reaction = None
        else:
            self.reaction = Reaction(node['children'][0], self)

    
class Reaction(object):
    def __init__(self, node, parent):
        
        self.smiles = getCanonicalReactionSmiles(node['smiles'])

        self.product = parent
        self.plausibility = node['plausibility']
        self.template_score = node['template_score']
        self.num_examples = node['num_examples']
        self.reactants = [Molecule(child, parent.depth+1) for child in node['children']]
        
        self.prd_rct_atom_cnt_diff = node.get('prd_rct_atom_cnt_diff', float('nan'))
        self.tf_id = node.get('tf_id', float('nan'))
        self.reaxys_id = node.get('reaxys_id', float('nan'))
        self.is_patent = node.get('is_patent', float('nan'))


def getCanonicalSmiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

  
def getCanonicalReactionSmiles(reactionSmiles):
    parts = reactionSmiles.split('>>')
    
    reactants = parts[0].split('.')
    products = parts[1].split('.')

    reactants = [getCanonicalSmiles(molecule) for molecule in reactants]
    products = [getCanonicalSmiles(molecule) for molecule in products]
    
    reactants.sort()
    products.sort()

    result = '{}>>{}'.format('.'.join(reactants), '.'.join(products))
    
    return result


def routeToTree(route):
    return Molecule(route)


def getReactions(rootMolecule):
    allReactions = []
    
    def extractReactions(molecule):
        if not molecule.is_terminal:
            allReactions.append(molecule.reaction)
            
            for reactant in molecule.reaction.reactants:
                extractReactions(reactant)
    
    extractReactions(rootMolecule)
    return allReactions


def getStartingMolecules(rootMolecule):
    allStartingMolecules = []

    def extractStartingMolecules(molecule):
        if molecule.is_terminal:
            allStartingMolecules.append(molecule)
            
        else:
            for reactant in molecule.reaction.reactants:
                extractStartingMolecules(reactant)
    
    extractStartingMolecules(rootMolecule)
    return allStartingMolecules
