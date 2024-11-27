import time
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import rdChemReactions
from rdkit import RDLogger
from src.utils import remove_isotopes_from_smiles
from src.utils import routeToTree

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def k_rxn_rct(smiles: str,
              max_branch: int = 50,
              filter_threshold: float = 0.5,
              rp_model=None,
              ff_model=None,
              templates: list = [],
              rxn_from_root: list = [],
              top_k=800,
              debug=0):
    """
    k_rxn_rct takes a smiles and predicts the next max_branch possible
    reactions, applies them and checks them with the given fast-filter
    model.

    Args:
        smiles (str): SMILES that prediction and application of
        templates should be done for
        max_branch (int): the branching factor of our search: "Maximum
        amount of reactions that should be tried for a given tree level."
        filter_threshold (float): Decision boundary for the fast
        filter model
        rp_model ([type]): reaction prediction model
        ff_model ([type]): fast filter model
        templates (list): templates corresponding to the model
        top_k (int, optional): threshold for the top_k reactions to
        use for route search. Defaults to 800.

    Returns:
        k_rxns_new (dictionary): dictionary of new reactions and the
        evaluation scores
    """
    scores = rp_model.predict([smiles])
    # sorted numpy array of the numbers corresponding to best templates
    best_temp = np.argsort(-scores[0])[:top_k]
    sorted_pred = scores[0][best_temp]

    # initialize a dict for each tf possibly considered for the actual target
    # right now, there is now way to now which one will actually survive the ff model
    # and become part of the route search. Therefore, they get all initialized with an empty list
    k_rxns = {}
    k_rxn = 0
    product_identifier = 0
    while ((len(k_rxns) < max_branch and sorted_pred[k_rxn] > 0.0001) or
           (len(k_rxns) < 25 and sorted_pred[k_rxn] > 0.00000001)) and (k_rxn < top_k-1):
        reaction_smarts = templates[best_temp[k_rxn]]["reaction_smarts"]
        reaction_smarts = '(' + reaction_smarts.replace('>>', ')>>(') + ')'
        rxn_obj = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        products = rxn_obj.RunReactants([Chem.MolFromSmiles(smiles)])
        if products != ():
            for product in products:
                product_sm = Chem.MolToSmiles(product[0], True)  # check here for the recursion error with [0][0]
                product_sm = product_sm.replace('/', '').replace('\\', '')
                product_sep = product_sm.split('.')
                # if our set of products is not empty, i.e. plausible, and it is not the
                # same as a set we got from another reaction-template before, we add it
                # to our dictionary of reactions for the molecule from smiles
                if tuple(product_sep) not in (tuple(k_rxns[rxn]["smiles_list"]) for rxn in k_rxns):

                    k_rxns["{}_{}".format(smiles, product_identifier)] = {"template": best_temp[k_rxn],
                                                                          "probability": sorted_pred[k_rxn],
                                                                          "rxn_class": templates[best_temp[k_rxn]]["rxn_class"],
                                                                          "rxn_class_id": templates[best_temp[k_rxn]]["rxn_class_id"],
                                                                          "smiles_list": product_sep,
                                                                          "smiles": product_sm}
                    product_identifier += 1
        k_rxn += 1
    fp_list = []
    key_list = []
    for rxn in k_rxns.keys():
        fp_tmp = ff_model.gen_fp(k_rxns[rxn]["smiles"], smiles)
        if isinstance(fp_tmp, np.ndarray):
            fp_list.append(fp_tmp)
            key_list.append(rxn)
    if fp_list:
        score = ff_model.predict(np.vstack(fp_list))
        bouncer = score > filter_threshold
    else:
        score = []
        bouncer = []
    k_rxns_new = {}
    for i in range(len(bouncer)):
        if bouncer[i]:
            k_rxns_new[key_list[i]] = k_rxns[key_list[i]]
            k_rxns_new[key_list[i]]["ff_score"] = float(score[i])

    return k_rxns_new


def look_up(smi, buyables):
    """check whether a molecule is buyable

    Args:
        smi (str): SMILES of targets to check
        buyables (dict): dictionary of all the building blocks

    Returns:
        bool: True if buyable, false if not
    """
    # here a list of excluded targets could be implemented and return false
    clean_smi = remove_isotopes_from_smiles(smi)
    if clean_smi in buyables.keys():
        return True
    else:
        return


def add_reaction(reaction, in_dict, levelm=0, levelr=0):
    """adds a reaction to the route dictionary. Some of the elements have no specific
     meaning in DFPNE and get only set to make the resulting structure compatible with the ASKCOS output structure.
    Arbitrarily set elements: prd_rct_atom_cnt_diff, num_examples, necessary_reagent, id

    Args:
        reaction (str): to the reaction assigned key in the input dictionary (tree from DFPNE)
        in_dict (dict): tree dictionary from the DFPNE algorithm encoding one route
        levelm (int): recursion level of the last molecule
        levelr (int): recursion level of the last reaction

    Returns:
        dict: dictionary which gets added to the final route
    """
    out_dict = {'plausibility': in_dict[reaction]['ff_score'],
                'template_score': in_dict[reaction]['probability'],
                'prd_rct_atom_cnt_diff': None,
                'tf_id': in_dict[reaction]['template'],
                'rxn_class': in_dict[reaction]['rxn_class'],
                'rxn_class_id': in_dict[reaction]['rxn_class_id'],
                'num_examples': 5,
                'necessary_reagent': '',
                'id': 0,
                'is_reaction': True,
                'children': []}
    if in_dict[reaction]['reactants']:
        for smi in in_dict[reaction]['reactants']:
            out_dict['children'].append(add_molecule(smi, in_dict, levelm, levelr))
    else:
        print("\nThis should never appear!!! A reaction without reactants doesn't make sense\n")
    out_dict['smiles'] = '{}>>{}'.format('.'.join(in_dict[reaction]['reactants']), reaction)
    return out_dict


def add_molecule(smiles, in_dict, levelm=0, levelr=0):
    """adds a chemical to the route dictionary. Some of the elements
    have no specific meaning in DFPNE and get only set to make the
    resulting structure compatible with the ASKCOS output structure.
    Arbitrarily set elements: ppg, as_reactant, as_product, id

    Args:
        smiles (str): SMILES of the chemical getting added to the dictionary (also key in input dictionary)
        in_dict (dict): tree dictionary from the DFPNE algorithm encoding one route
        levelm (int): recursion level of the last molecule
        levelr (int): recursion level of the last reaction

    Returns:
        dict: dictionary which gets added to the final route
    """
    out_dict = {'smiles': smiles,
                "ppg": 0.0,
                "as_reactant": 0,
                "as_product": 0,
                "id": 0,
                "is_chemical": True}
    if in_dict[smiles]['reactants']:
        out_dict['children'] = [add_reaction(smiles, in_dict, levelm, levelr)]
    else:
        out_dict['children'] = []
    return out_dict


def evalTree(tree, target):
    """converts the trees dictionary from DFPNE-algorithm into ASKCOS compatible form

    Args:
        tree (dict): dictionary form the DFPNE-algorithm containing the last found tree
        target (str): SMILES of the target molecule

    Returns:
        trees_list (list): Route dictionaries

    """
    tree_dict = add_molecule(target, tree)
    tree = routeToTree(tree_dict)

    return tree_dict, 
