import os
import glob
import pickle
import numpy as np
import re
import argparse
import rdkit.Chem as Chem

from src.utils import routeToTree
from src.utils import getStartingMolecules


# auxiliary functions to sort the filenames naturally
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def load_routes(filename):
    with open(filename, 'rb') as infile:
        routes = pickle.load(infile)
    return routes


def removeIsotopes_from_mol(mol):
    """ Removes the isotope label of each atom of a SMILES.
    If the provided smiles string is not canonical it will be after
    the removal.

    Parameters
    ----------
    mol : object
        A rdkit molecule object.
    Returns
    -------
    object
        A rdkit molecule object without isotopes.
    """
    mol_tmp = Chem.Mol(mol)
    atom_data = [(atom, atom.GetIsotope()) for atom in mol_tmp.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            atom.SetIsotope(0)
    return mol_tmp


def recursive_items(dictionary):
    for key, value in dictionary.items():
        if key == "smiles":
            rr_list = []
            for rr in value.split(">>"):
                rr_list.append(removeIsotopes_from_mol(rr))
            value = ">>".join(rr_list)
            dictionary[key] = value
        if type(value) is dict:
            yield from recursive_items(value)
        elif type(value) is list:
            for itm in value:
                if type(itm) is dict:
                    yield from recursive_items(itm)
        else:
            yield (key, value)


def analyze_disconn(routes, target_bonds):
    """ Analyze the disconnections of a route and count how often each bond gets broken.

    Parameters
    ----------
    routes : list
        A list of routes where every route is given as a dictionary
    target_bonds : dict
        A dict of all bonds of the target molecule

    Returns
    -------
    list_of_disconnections : list
        A list of arrays where each array represents the disconnections of a route
    """

    list_of_disconnections = []
    for idx, route in enumerate(routes):
        routeTree = routeToTree(route)
        molecules = getStartingMolecules(routeTree)
        smiles = [mol.smiles for mol in molecules]

        # Get a list of all bonds in all starting materials of the currently observed route
        reactants_bonds = []
        for reactant in smiles:
            r_mol = Chem.MolFromSmiles(reactant)
            for bond in r_mol.GetBonds():
                begin_atom_iso, _ = bond.GetBeginAtom().GetIsotope(), bond.GetBeginAtomIdx()
                end_atom_iso, _ = bond.GetEndAtom().GetIsotope(), bond.GetEndAtomIdx()
                reactants_bonds.append(tuple(sorted([begin_atom_iso, end_atom_iso])))

        # count how often each bond gets broken in the currently observed route
        disconn_in_route = np.zeros((len(target_bonds.keys()))).astype(int)
        for bb in target_bonds.keys():
            if target_bonds[bb] not in reactants_bonds:
                disconn_in_route[bb] = 1

        # collect all disconnections appearing in this route
        list_of_disconnections.append(disconn_in_route)

    return list_of_disconnections


def get_bonds_list(smi):
    mol = Chem.MolFromSmiles(smi)
    bonds_dict = {}
    for bond_idx in range(len(mol.GetBonds())):
        bond = mol.GetBondWithIdx(bond_idx)
        begin_atom_iso, _ = bond.GetBeginAtom().GetIsotope(), bond.GetBeginAtomIdx()
        end_atom_iso, _ = bond.GetEndAtom().GetIsotope(), bond.GetEndAtomIdx()
        bonds_dict[bond_idx] = tuple(sorted([begin_atom_iso, end_atom_iso]))
    return bonds_dict


def calculate_diversity_score(list_of_sets: list):
    if len(list_of_sets) > 1:
        score = 0.0
        for ii in range(0, len(list_of_sets)):
            frac = 1.
            for jj in range(0, len(list_of_sets)):
                if ii == jj:
                    continue
                sum_or_vec = sum(np.bitwise_or(list_of_sets[ii], list_of_sets[jj]))
                sum_and_vec = sum(np.bitwise_and(list_of_sets[ii], list_of_sets[jj]))
                diff = sum_or_vec - sum_and_vec
                frac += (diff / sum_or_vec)
            score += frac
        score = score / len(list_of_sets)
    elif len(list_of_sets) == 1:
        score = 1.0
    else:
        score = 0.0
    return score


def get_parent_sets(list_of_arrays):
    unique_disconn_list = []
    for ii in range(len(list_of_arrays)):
        ii_array = list_of_arrays[ii]
        ii_is_parent = True
        for jj in range(len(list_of_arrays)):
            jj_array = list_of_arrays[jj]
            if ii == jj:
                continue
            if np.array_equal(np.bitwise_and(ii_array, jj_array), jj_array):
                ii_is_parent = False
        if ii_is_parent:
            unique_disconn_list.append(list_of_arrays[ii])
    return unique_disconn_list


def main(base_dir, outfile, verbosity=0):

    mol_dir = os.path.join(base_dir, "*.pkl")
    molecules_files = glob.glob(mol_dir)
    if not os.path.exists(outfile):
        with open(outfile, 'w') as logFile:
            logFile.write('index,num_routes,num_unique_sets,num_parent_sets,diversity_score\n')
    results = []

    with open(outfile, 'a') as logFile:
        counter = 0
        for mol_file in molecules_files:
            mol_str = mol_file.split("/")[-1].split(".")[0]
            mol_id = mol_str.split("_")[-2]
            print(mol_id)

            routes = load_routes(mol_file)
            if verbosity > 0:
                print("number of routes: {}".format(len(routes)))

            if len(routes) == 0:
                logFile.write('{},{},{},{},{}\n'.format(mol_id, len(routes), 0, 0, 0.))
                logFile.flush()
                results_dict = {"mol_id": mol_id,
                                "num_routes": len(routes),
                                "uniq_tuples": 0,
                                "num_parent_sets": 0,
                                "diversity_score": 0.}
                results.append(results_dict)
                continue

            if routes == "Target is already buyable.":
                counter += 1
                if verbosity > 0:
                    print("Counter: ", counter)
                continue
            target_smiles = routeToTree(routes[0]).smiles
            target_bonds = get_bonds_list(target_smiles)

            # get the disconnections and statistics on how often the individual bonds get broken
            list_of_disconn_per_route = analyze_disconn(routes, target_bonds)
            # get unique sets
            unique_arrays_tmp = set(map(tuple, list_of_disconn_per_route))
            unique_arrays_list = [np.array(aa) for aa in unique_arrays_tmp]
            # get parent sets (those which cannot be represented by another set)
            parent_sets_list = get_parent_sets(unique_arrays_list)
            # calculate the diversity score (can be interpreted as number of truly different routes)
            diversity_score = calculate_diversity_score(parent_sets_list)

            if verbosity > 0:
                print("number of unique tuples:  ", len(unique_arrays_list))
                print("number of parent sets:", len(parent_sets_list))
                print("diversity_score", diversity_score)

            logFile.write('{},{},{},{},{}\n'.format(mol_id,
                                                    len(routes),
                                                    len(unique_arrays_list),
                                                    len(parent_sets_list),
                                                    diversity_score))
            results_dict = {"mol_id": mol_id,
                            "num_routes": len(routes),
                            "uniq_tuples": len(unique_arrays_list),
                            "num_parent_sets": len(parent_sets_list),
                            "diversity_score": diversity_score}
            results.append(results_dict)
            logFile.flush()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default="/data/glosu/dfpns/results/benchmark_set/")
    parser.add_argument('-o', '--outfile', default="route_analyse.csv")
    args = parser.parse_args()

    input_dir = str(args.input_dir)
    outfile = str(args.outfile)

    results = main(input_dir, outfile)
    print(results)
