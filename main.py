from math import log
from src.DFPNE_retro import RouteSearch
from src.utils import load_data
from src.models import RpFingerprintModel
from src.models import FfFingerprintModel
import rdkit.Chem as Chem

import time
import sys
import os
import traceback
import pickle
import argparse
import random
import pandas as pd
import numpy as np
import faulthandler
import concurrent.futures  # for code parallelization


def search(molecule):
    rp_model_path = '/path/to/rp_model.{pt|h5}'
    variance_path = '/path/to/rp_model_variance.{pt|h5}'
    template_path = '/path/to/rp_templates.json.gz'
    ff_model_path = '/path/to/ff_model.{pt|h5}'
    buyables_path = '/path/to/building_blocks.json.gz'

    # load templates and buyables
    templates, buyables = load_data(template_path,
                                    buyables_path)

    # load the reaction prediction model
    rp_model = RpFingerprintModel(rp_model_path=rp_model_path,
                                      variance_path=variance_path)

    # load the fast filter model
    ff_model = FfFingerprintModel(ff_model_path=ff_model_path)

    idx, id_column, mol = molecule["idx"], molecule["id_column"], molecule["mol"]

    Chem.rdmolops.RemoveStereochemistry(mol)
    smiles = Chem.MolToSmiles(mol, True)
    if verbose > 0:
        print('Target molecule (canonicalized, no stereochemistry): {}'.format(smiles))

    if os.path.exists('{}_{}_{}.pkl'.format(outfile, idx, id_column)) and (args.overwrite is False):
        print('Skipping {}: {}'.format(id_column, smiles))
        return

    start_time = time.time()
    try:
        routeSearch = RouteSearch(rp_model=rp_model,
                                  ff_model=ff_model,
                                  templates=templates,
                                  buyables=buyables,
                                  start_time=start_time,
                                  target=smiles,
                                  max_routes=max_trees,
                                  max_search_time=expansion_time,
                                  sigma=sigma,
                                  mult_paths_penalty=penalty,
                                  max_depth=max_depth,
                                  max_branch=max_branching,
                                  filter_threshold=filter_threshold,
                                  verbose=verbose)

        routes, status = routeSearch.DFPN()
        if status == 1:
            if verbose > 0:
                print('Search time ({} sec.) is up for: {}'.format(expansion_time, smiles))

    except BaseException as err:
        track = traceback.format_exc()
        print(track)
        print('Exception when calling DFPN: {}'.format(err))
        print('The Molecule was: {}'.format(smiles))
        status = -1
        routes = []

    end_time = time.time()

    logFile.write('{},{},{},{},{}\n'.format(id_column,
                                            smiles,
                                            "\"" + str(status) + "\"",
                                            len(routes),
                                            end_time - start_time))
    logFile.flush()
    if verbose > 0:
        print("routeSearch exited with status: {}".format(status))
        print('Total num routes: {}'.format(len(routes)))
    with open('{}_{}_{}.pkl'.format(outfile, idx, id_column), 'wb') as outputFile:
        pickle.dump(routes, outputFile)


if __name__ == "__main__":

    faulthandler.enable()

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--expansion_time', default=300)
    parser.add_argument('-s', '--seed', default=42)
    parser.add_argument('-i', '--infile', default='input.smi')
    parser.add_argument('-o', '--outfile', default='results')
    parser.add_argument('-m', '--max_trees', default=500)
    parser.add_argument('-f', '--filter', default=0.5)
    parser.add_argument('-d', '--max_depth', default=6)
    parser.add_argument('-b', '--max_branching', default=50)
    parser.add_argument('-p', '--penalty', default=10)
    parser.add_argument('-sig', '--sigma', default=3)
    parser.add_argument('-id_col', '--id_column', default="ID")
    parser.add_argument('-ow', '--overwrite', default=False)
    parser.add_argument('-v', '--verbose', default=0)
    parser.add_argument('-n', '--num_proc', default=4)
    args = parser.parse_args()

    expansion_time = int(args.expansion_time)
    seed = int(args.seed)
    infile = args.infile
    outfile = args.outfile
    max_trees = int(args.max_trees)
    filter_threshold = float(args.filter)
    max_depth = int(args.max_depth)
    max_branching = int(args.max_branching)
    penalty = int(args.penalty)
    sigma = int(args.sigma)
    verbose = int(args.verbose)
    num_proc = int(args.num_proc)

    random.seed(seed)
    np.random.seed(seed)

    logPath = outfile + '_log.txt'
    if not os.path.exists(os.path.dirname(logPath)):
        try:
            os.makedirs(os.path.dirname(logPath))
        except OSError as exc:
            print("Could not make output directory.")
            print(exc)
            sys.exit()
    if args.overwrite is False:
        with open(logPath, 'w') as logFile:
            logFile.write('Index,SMILES,Status,Routes,Time\n')

    with open(logPath, 'a') as logFile:

        df_in = pd.read_csv(infile)
        if verbose > 0:
            print("working on {} molecule(s)".format(len(df_in)))

        # list of dictionaries containing the information needed to start the search function, since map needs a list
        molecules = []
        for idx, row in df_in.iterrows():
            tmp_dict = {"idx": idx, "id_column": row[args.id_column], "mol": Chem.MolFromSmiles(row['SMILES'])}
            molecules.append(tmp_dict)
        if num_proc == 1:  # serial case
            for m in molecules:
                search(m)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(molecules), num_proc)) as executor:
                executor.map(search, molecules)
