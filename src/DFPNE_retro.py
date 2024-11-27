# DFPN_retro, but with the ability to give out multiple routes

import math
from src.EdgeGeneration import look_up
from src.EdgeGeneration import evalTree
from src.EdgeGeneration import k_rxn_rct
import rdkit.Chem as Chem
import time
import sys

# DFPNS is recursive -> default system setting is at 1000
sys.setrecursionlimit(10 ** 6)


class RouteSearch:

    def __init__(self,
                 rp_model,
                 ff_model,
                 templates,
                 buyables,
                 start_time,
                 g=None,
                 target: str = 'CC(=O)OC1=CC=CC=C1C(=O)O',
                 max_routes: int = math.inf,
                 max_search_time: int = 60,
                 sigma: int = 3,
                 mult_paths_penalty: int = 5,
                 max_depth: int = 6,
                 max_branch: int = 50,
                 filter_threshold: float = 0.5,
                 labeling: bool = True,
                 verbose: int = 0):

        if g is None:
            g = {}
        self.graph = g
        self.rp_model = rp_model
        self.ff_model = ff_model
        self.templates = templates
        self.buyables = buyables
        self.start_time = start_time
        if labeling:
            tmp_mol = Chem.MolFromSmiles(target)
            for atm in tmp_mol.GetAtoms():
                atm.SetIsotope(1 + atm.GetIdx())
            self.target = Chem.MolToSmiles(tmp_mol, True)
        else:
            self.target = Chem.MolToSmiles(Chem.MolFromSmiles(target), True)
        self.verbose = verbose
        self.max_routes = max_routes
        self.max_search_time = max_search_time
        self.mult_paths_penalty = mult_paths_penalty
        self.max_depth = max_depth
        self.max_branch = max_branch
        self.filter_threshold = filter_threshold
        # sigma is a threshold controlling parameter as in the DFPN-E paper
        # it is used to avoid re-expansions at the cost of possibly
        # making the direction of search less precise
        # sensible values for sigma: 1 - 15
        self.sigma = sigma
        # node_info = {node: {phi, delta, dp_depth, dp_paths, min_dist}} where dp_paths is  a list of all the paths for
        # which this reaction should be considered disproven and md is the minimal distance for a molecule to r, and
        # irrelevant for reactions
        # phi: proof number for OR nodes disprove number for AND nodes
        # delta: proof number for AND nodes disprove number for OR nodes
        # dp_depth: at which depth a node is disproven
        # dp_paths: a list of all the paths for which this reaction should be considered disproven
        # min_dist: minimal distance for a molecule to r (depth - 1, how many edges). Irrelevant for reactions
        self.node_info = {}  # initialize node_info as an empty dictionary
        # path from the root to the current molecule
        self.path_from_root = []
        # similar to path_from_root but tracks the rxn classes used in the path
        self.rxn_from_root = []
        # chai_trees stores all route trees that are found by the algorithm in CHAi's accepted format
        self.chai_trees = []
        # proof_trees stores all route trees in an easier to read format (debug purposes, since chai_trees is confusing)
        self.proof_trees = {}
        # initialized for the build_tree function, needed for multiple routes
        self.mult_paths = None
        # how many routes have we found
        self.found = 0
        self.calls = 0  # counter for the number of calls to the one-step model
        # self.proof_depth = 0
        # inc_flag: flag for the TCA, if True the TCA increases thresholds until expanding
        # a leaf or otherwise changing the searched tree (by e.g. disproving a reaction)
        # (TCA: threshold controlling algorithm) (must be false to begin with, gets set true in algorithm)
        self.inc_flag = False
        self.no_rxn_flag = False
        self.last_expand = time.time()  # if a random loop happens

    def DFPN(self, level=0):
        """
        DFPN recursively guides the algorithm and initializes the target molecule

        Args:
            level (int): the recursion level the algorithm is currently on
        """
        # initializing the thresholds of root node r with infinity
        th_phi_r = math.inf
        th_delta_r = math.inf
        # initializing our target molecule
        self.node_info[self.target] = {"phi": 0,
                                       "delta": 0,
                                       "dp_depth": math.inf,
                                       "dp_paths": [],
                                       "min_dist": 0,
                                       }
        # path_from_root is the path we are currently on, it is needed
        # to check for directed cycles (in select_child_AND)
        self.path_from_root.append(self.target)
        status = self.or_search(self.target, th_phi_r, th_delta_r)
        if status == 1:
            return self.chai_trees, status
        phi_r = self.node_info[self.target]["phi"]
        delta_r = self.node_info[self.target]["delta"]
        try:
            if phi_r == 0:
                tree, _ = self.build_tree(self.target)
                # if build_tree gives back an empty dictionary, the algorithm wrongly stopped and needs to start again
                if not tree:
                    # print("Empty tree!")
                    return self.DFPN(level=level + 1)
                too_deep = False
                for mol in tree:
                    if tree[mol]["depth"] > self.max_depth + 1:
                        too_deep = True
                if not too_deep:
                    self.found += 1
                    if self.verbose > 0:
                        print('Found: ', self.found)
                    self.proof_trees[self.found] = tree
                    route_for_chai = evalTree(tree, self.target)
                    self.chai_trees.append(route_for_chai)

                    if self.found >= self.max_routes or time.time() - self.start_time > self.max_search_time:
                        if self.verbose > 0:
                            print("Number of calls to the one-step model:", self.calls)
                        return self.chai_trees, status
                    else:
                        self.updateNodeInfo()
                        return self.DFPN(level=level + 1)
                else:
                    self.updateNodeInfo()
                    return self.DFPN(level=level + 1)
            elif delta_r == 0:
                if self.verbose > 0:
                    print("\nFound", str(len(self.proof_trees)), "routes.")
                return self.chai_trees, status
            else:
                status = 2
                return self.chai_trees, status
        except KeyError:
            return self.chai_trees, status

    def or_search(self, n, th_phi_n, th_delta_n):
        """
        or_search guides the search for the most promising reaction of given molecule n (called MPN in DFPNS).
        This MPN gets then searched by calling the function and_search.
        At the end values like proof- and disproof number get updated.

        Args:
            n (str): molecule in smiles format
            th_phi_n (Union[int, float]): threshold for the proof number of n
            th_delta_n (Union[int, float]): threshold for the disproof number of n
        """
        # check if molecule n has already been proven or disproven
        if (self.node_info[n]["phi"] == math.inf) or (self.node_info[n]["delta"] == math.inf):
            # every time we leave a search, we need to remove our top node from path_from_root, as it leaves the path
            self.path_from_root.remove(n)
            return 0
        # check if we have to expand molecule n
        if n not in self.graph.keys():
            self.last_expand = time.time()
            self.expand(n)
            # flag for the TCA gets set to False after an expansion (in most cases self.inc_flag was already False)
            self.inc_flag = False
        # update md if the current distance is shorter than its value
        self.node_info[n]["min_dist"] = min(self.node_info[n]["min_dist"], len(self.path_from_root) - 1)
        # deltaMin(n) and self.sum_phi(n) represent phi_n and delta_n
        # they need to be smaller than their thresholds for our depth first variant of PNS
        while (self.deltaMin_OR(n) < th_phi_n) and (self.sum_phi(n) < th_delta_n):
            n1, phi1, delta2 = self.select_child_OR(n)
            th_phi_n1 = th_delta_n + phi1 - self.sum_phi(n)
            th_delta_n1 = min(th_phi_n, delta2 + self.sigma) - self.h(n, n1)
            self.rxn_from_root.append(self.node_info[n1]["rxn_class_id"])
            status = self.and_search(n1, th_phi_n1, th_delta_n1, self.node_info[n]["min_dist"])
            if status == 1:
                return 1
        # after we looked at every child node, we can update the values for node n in our table/dict
        self.node_info[n].update({"phi": self.deltaMin_OR(n),
                                  "delta": self.sum_phi(n)})
        # checking if the molecule is considered disproven because of maximum depth
        if (self.node_info[n]["phi"] == math.inf) and (self.node_info[n]["dp_depth"] > len(self.path_from_root)):
            # if at least one of its reactions is disproven because of max_depth, the molecule has the same property
            # that is because we only need for example this one reaction to prove the molecule, so at a smaller depth
            # the reaction and therefore the molecule might still be provable
            # this is also the reason, why the reaction with the maximum dp_depth sets the dp_depth for the molecule
            depth = 0
            for i in self.graph[n]:
                if self.node_info[i]["phi"] == 0:
                    depth = max(self.node_info[i]["dp_depth"], depth)  # as said, we need the rxn with the max dp_depth
                    # if node_info[n]["phi"] indicates n is disproven, it's because every one of its rxns is, if one of
                    # these reactions is dp because of a loop, the disproof_depth might not be the current depth
                    # afterwards, but the saved disproven path indicates the disproof_depth
                    if (self.path_from_root in self.node_info[i]["dp_paths"]) and \
                            (self.path_from_root not in self.node_info[n]["dp_paths"]):
                        self.node_info[n]["dp_paths"].append(self.path_from_root[:])
            # the disproven depth should never get higher than it was before, so we take the minimum
            self.node_info[n]["dp_depth"] = min(self.node_info[n]["dp_depth"], depth)
        self.path_from_root.remove(n)
        if (time.time() - self.start_time > self.max_search_time):
            return 1
        return 0

    def and_search(self, n, th_phi_n, th_delta_n, md_parent):
        """
        and_search guides the search for the most promising molecule of given reaction n (called MPN in DFPNS).
        This MPN gets then searched by calling the function and_search.
        At the end values like proof- and disproof number get updated.

        Args:
            n (str): reaction in smiles format
            th_phi_n (Union[int, float]): threshold for the disproof number of n
            th_delta_n (Union[int, float]): threshold for the proof number of n
            md_parent (int): minimal distance of the parent molecule to reaction n
        """
        # If we get too deep and our reaction is not proven, we at first consider it to be disproven, however we
        # save the current depth of the reaction for its potential usage at a smaller depth
        if len(self.path_from_root) + 1 > self.max_depth:
            for i in self.graph[n]:
                if self.node_info[i]["phi"] != 0:
                    depth = min(self.node_info[n]["dp_depth"], len(self.path_from_root))
                    self.node_info[n].update({"phi": 0,
                                              "delta": math.inf,
                                              "dp_depth": depth})
                    self.inc_flag = False  # inc_flag should be False again, since the tree changed
                    # remove reaction from list at role back (reverse to ensure we remove the latest occurrence since
                    # rxn classes can occur multiple times)
                    self.rxn_from_root.reverse()
                    self.rxn_from_root.remove(self.node_info[n]["rxn_class_id"])
                    self.rxn_from_root.reverse()
                    return 0
        for i in self.graph[n]:  # the check for an unproven 'old' child node, combating the overestimation error
            if (md_parent >= self.node_info[i]["min_dist"]) and (self.node_info[i]["phi"] != 0):
                self.inc_flag = True
                break
        if self.inc_flag:  # increase the thresholds if the TCA detected an unproven old child node
            th_phi_n = max(th_phi_n, self.deltaMin_AND(n) + 1)
            th_delta_n = max(th_delta_n, self.sum_phi(n) + 1)
        # if the AND-node n only has one child node, we can skip the selectChild-fct., also the thresholds stay the same
        if len(self.graph[n]) == 1:
            n1 = self.graph[n][0]
            self.path_from_root.append(n1)  # add the new best child node to the path_from_root
            th_phi_n1 = th_delta_n
            th_delta_n1 = th_phi_n
            status = self.or_search(n1, th_phi_n1, th_delta_n1)
            if status == 1:
                return 1
        else:
            while (self.deltaMin_AND(n) < th_phi_n) and (self.sum_phi(n) < th_delta_n):
                n1, phi1, delta2 = self.select_child_AND(n)
                self.path_from_root.append(n1)  # add the new best child node to the path_from_root
                th_phi_n1 = th_delta_n + phi1 - self.sum_phi(n)
                th_delta_n1 = min(th_phi_n, delta2 + 1)
                status = self.or_search(n1, th_phi_n1, th_delta_n1)
                if status == 1:
                    return 1
        # after we looked at every child node, we can update the values for node n in our table/dict
        self.node_info[n].update({"phi": self.deltaMin_AND(n),
                                  "delta": self.sum_phi(n)})
        # checking if the reaction is considered disproven because of maximum depth
        if (self.node_info[n]["phi"] == 0) and (self.node_info[n]["dp_depth"] > len(self.path_from_root)):
            # if all of its disproven child nodes are disproven because of max_depth, the reaction has the same property
            # that is because for a reaction every molecule is needed to prove it, if one molecule is disproven at every
            # depth, the dp_depths of the other targets don't matter
            # this is also the reason, why the molecule with the minimum dp_depth sets the dp_depth for the molecule
            depth = len(self.path_from_root) + 1  # could also be math.inf
            # i iterates over targets
            for i in self.graph[n]:
                if self.node_info[i]["phi"] == math.inf:  # mol is disproven
                    depth = min(self.node_info[i]["dp_depth"], depth)  # as said, we need the mol with the min dp_depth
                    node_path = self.path_from_root[:]
                    node_path.append(i)  # node_path is the path to and including mol i
                    # if node_path is disproven for i, since a rxn is associated with its parent node, the path without i
                    # should be disproven for the rxn leading to i, so we append it to the dp_paths of i (if needed)
                    if (node_path in self.node_info[i]["dp_paths"]) and \
                            (self.path_from_root[:] not in self.node_info[n]["dp_paths"]):
                        self.node_info[n]["dp_paths"].append(self.path_from_root[:])
            # we have to use depth-1, because the depth of a reaction is associated with its parent node and not its child nodes
            self.node_info[n]["dp_depth"] = min(self.node_info[n]["dp_depth"], depth - 1)
        # remove reaction from list at role back (reverse to ensure we remove the
        # latest occurrence since rxn classes can
        # occur multiple times)
        self.rxn_from_root.reverse()
        self.rxn_from_root.remove(self.node_info[n]["rxn_class_id"])
        self.rxn_from_root.reverse()
        return 0

    def expand(self, n):
        """
        expand expands and initializes the reactions of molecule smi and the corresponding targets needed for those
        reactions.

        Args:
            n (str): molecule in smiles format
        """
        self.calls += 1
        # get k(==max_branching) returns reactions and reactants for n
        rxns = k_rxn_rct(n,
                         max_branch=self.max_branch,
                         filter_threshold=self.filter_threshold,
                         rp_model=self.rp_model,
                         ff_model=self.ff_model,
                         templates=self.templates,
                         rxn_from_root=self.rxn_from_root,
                         debug=self.verbose)
        # initialize graph
        self.graph[n] = {}
        for i in rxns:
            self.graph[n][i] = rxns[i]  # assign probability (reaction prediction)
        # self.graph[n] is a dictionary of possible reactions for the molecule n
        # every reaction gets his own node in self.graph, therefore the loop goes over i in self.graph[n]:
        for i in self.graph[n]:
            self.graph[i] = rxns[i]["smiles_list"]
            disproof_depth_i = math.inf  # depth at which reaction i is disproven gets initialized with inf
            # every reactant of reaction i also gets a node in self.graph:
            for smi_count, smi in enumerate(self.graph[i]):
                if smi not in self.node_info.keys():
                    if look_up(smi, self.buyables) is True:
                        # initializing buyable molecule smi
                        self.node_info[smi] = {"phi": 0,
                                               "delta": math.inf,
                                               "dp_depth": math.inf,
                                               "dp_paths": [],
                                               "min_dist": len(self.path_from_root)}
                        self.graph[smi] = True
                    else:
                        # initializing non-buyable molecule smi
                        self.node_info[smi] = {"phi": 1,
                                               "delta": 1,
                                               "dp_depth": math.inf,
                                               "dp_paths": [],
                                               "min_dist": len(self.path_from_root)}
                else:
                    # a reaction is considered disproven at the minimum disproof_depth of its child nodes - 1
                    # for explanation why see end of and_search
                    disproof_depth_i = min(disproof_depth_i, self.node_info[smi]["dp_depth"] - 1)
                    # sometimes smi can be the target, if so the reaction should be disproven forever (i.e. at depth -1)
                    if smi == self.target:
                        disproof_depth_i = -1
                    # if min_dist_smi > len(path_from_root) the min_dist_smi needs to be updated
                    self.node_info[smi]["min_dist"] = min(self.node_info[smi]["min_dist"], len(self.path_from_root))
            self.node_info[i] = {"phi": self.deltaMin_AND(i),
                                 "delta": self.sum_phi(i),
                                 "dp_depth": disproof_depth_i,
                                 "dp_paths": [],
                                 "rxn_class": rxns[i]["rxn_class"],
                                 "rxn_class_id": rxns[i]["rxn_class_id"]}
        return

    def select_child_OR(self, smi):
        """
        select_child_OR selects the reaction of molecule smi that is the MPN (Most Proving Node) by searching for the
        two smallest delta-values of child nodes of smi. However here we need to add the heuristic function to the delta
        value first before finding the two smallest deltas. The second smallest delta value then belongs to the second
        most promising child node of rxn.

        Args:
            smi (str): molecule in smiles format

        Returns:
            n1 (str): most promising child node of smi in smiles format
            phi1 (Union[int, float]): phi value of n1
            delta2 (Union[int, float]): delta value of the second most promising child node
        """
        n1 = None
        phi1 = math.inf
        delta1 = math.inf
        delta2 = math.inf
        # rxn iterates over the reactions of molecule smi
        for rxn in self.graph[smi]:
            h_smi_rxn = self.h(smi, rxn)  # heuristic edge initialization
            if self.node_info[rxn]["delta"] != math.inf:  # allows us to skip child nodes that are considered disproven
                # the smallest (delta + h) dictates the most promising node
                if self.node_info[rxn]["delta"] + h_smi_rxn < delta1:
                    n1 = rxn
                    phi1 = self.node_info[rxn]["phi"]
                    delta2 = delta1
                    delta1 = self.node_info[rxn]["delta"] + h_smi_rxn
                    # the second smallest (delta + h) dictates the second most promising node
                elif self.node_info[rxn]["delta"] + h_smi_rxn < delta2:
                    delta2 = self.node_info[rxn]["delta"] + h_smi_rxn
                # if we have looked at at least 2 child nodes and delta1 can't get smaller, we can return earlier
                if (delta1 == 0) and (delta2 != math.inf):
                    return n1, phi1, delta2
        return n1, phi1, delta2

    def select_child_AND(self, rxn):
        """
        select_child_AND selects the reaction of reaction rxn that is the MPN (Most Proving Node) by searching for the
        two smallest delta-values of child nodes of rxn. The second smallest delta value then belongs to the second most
        promising child node of rxn.

        Args:
            rxn (str): reaction in smiles format

        Returns:
            n1 (str): most promising child node of rxn in smiles format
            phi1 (Union[int, float]): phi value of n1
            delta2 (Union[int, float]): delta value of the second most promising child node
        """
        n1 = None
        phi1 = math.inf
        delta1 = math.inf
        delta2 = math.inf
        for i in self.graph[rxn]:
            # the smallest delta dictates the most promising node
            if self.node_info[i]["delta"] < delta1:
                n1 = i
                phi1 = self.node_info[i]["phi"]
                delta2 = delta1
                delta1 = self.node_info[i]["delta"]
            # the second smallest delta dictates the second most promising node
            elif self.node_info[i]["delta"] < delta2:
                delta2 = self.node_info[i]["delta"]
        return n1, phi1, delta2

    def deltaMin_OR(self, smi, update_node_info_flag=False):
        """
        deltaMin_OR calculates the minimum delta plus the incorporated heuristic function h(smi, rxn) of smi's child nodes.
        It also detects cycles by checking the current path and is sometimes used to re-expand reactions.

        Args:
            smi (str): molecule in smiles format
            update_node_info_flag (bool): flag indicating if deltaMin_or was called from the update_node_info function

        Returns:
            delta_min (Union[int, float]): minimum delta (plus h(smi, rxn)) of smi's child nodes
        """
        delta_min = math.inf
        for rxn in self.graph[smi]:
            # some rxns of smi may have gotten re-expanded, if so we need to check if their dp_depth < current depth or
            # if the current path is in their disproven paths, if so they should be considered disproven again
            if (len(self.path_from_root) > self.node_info[rxn]["dp_depth"]) or \
                    (self.path_from_root in self.node_info[rxn]["dp_paths"]):
                self.node_info[rxn].update({"phi": 0,
                                            "delta": math.inf})
                continue
            # check if one child node of AND-node rxn is in the path we are currently on, if so we set it to disproven
            # iterate over targets j from reaction rxn
            for j in self.graph[rxn]:
                if j in self.path_from_root:  # check if we have a cycle
                    if j == self.path_from_root[0]:  # if j is the target we don't want to ever re-expand rxn
                        self.node_info[rxn].update({"phi": 0,
                                                    "delta": math.inf,
                                                    "dp_depth": -1})
                        self.inc_flag = False  # inc_flag should be False again, since the tree changed
                        break
                    # to combat the GHI-Problem, we save every path for which a reaction is considered disproven
                    self.node_info[rxn].update({"phi": 0,
                                                "delta": math.inf})
                    self.node_info[rxn]["dp_paths"].append(self.path_from_root[:])
                    self.inc_flag = False  # inc_flag should be False again, since the tree changed
                    break
            # check if max_depth or a certain path was the cause for a disproof of some of smi's child nodes
            # if a reaction is disproven at depth 6 for reaching max depth, it still can be reused at depth 1
            if (self.node_info[rxn]["delta"] == math.inf) and \
                    (self.node_info[rxn]["dp_depth"] > len(self.path_from_root)) and \
                    (self.path_from_root not in self.node_info[rxn]["dp_paths"]):
                # we can then re-expand rxn by setting the pn/dn to values that would indicate rxn just got expanded
                self.node_info[rxn].update({"phi": 1,
                                            "delta": len(self.graph[rxn])})
            if self.node_info[rxn]["phi"] == math.inf:
                if update_node_info_flag:
                    self.node_info[rxn].update({"phi": self.deltaMin_AND(rxn),
                                                "delta": self.sum_phi(rxn)})
                    # if a reaction is proven, we found a usable reaction and the heuristic function, which is only
                    # supposed to guide the search, does not need to be added to delta_min
                    if self.node_info[rxn]["phi"] == math.inf:
                        return 0
                    else:
                        delta_min = min(delta_min, self.node_info[rxn]["delta"] + self.h(smi, rxn))
                else:
                    return 0
            else:
                delta_min = min(delta_min, self.node_info[rxn]["delta"] + self.h(smi, rxn))
        return delta_min

    def deltaMin_AND(self, rxn):
        """
        deltaMin_OR calculates the minimum delta of rxn's child nodes and is sometimes used to re-expand targets.

        Args:
            rxn (str): reaction in smiles format

        Returns:
            delta_min (Union[int, float]): minimum delta of rxn's child nodes
        """
        delta_min = math.inf
        for mol in self.graph[rxn]:
            # check if max_depth was the cause for a disproof of some of rxn's child nodes
            # if a molecule is disproven at depth 6 for reaching max depth, it still can be reused at depth 1
            if (self.node_info[mol]["phi"] == math.inf) and \
                    (self.node_info[mol]["dp_depth"] > len(self.path_from_root) + 1):
                node_path = self.path_from_root[:]
                node_path.append(mol)
                # need to check if mol is disproven at node_path, which is the current path_from_root with mol added to
                # it, as disproven paths get saved like that in dp_paths for a molecule mol
                if node_path not in self.node_info[mol]["dp_paths"]:
                    # we can then re-expand mol by setting the pn/dn to values that would indicate it just got expanded
                    self.node_info[mol].update({"phi": 1,
                                                "delta": len(self.graph[mol])})
            delta_min = min(delta_min, self.node_info[mol]["delta"])
        return delta_min

    def sum_phi(self, n):
        """
        sum_phi calculates the sum of all phi of n's child nodes and throws out reactions that use the target as a reactant

        Args:
            n (str): molecule or reaction in smiles format

        Returns:
            sum_phi_temp (Union[int, float]): sum of all phi of n's child nodes
        """
        sum_phi_temp = 0
        # targets that are necessary for reaction n or vice versa
        for i in self.graph[n]:
            if i == self.target:  # if n is a reaction and i is the target, the proof number obviously should be infinity
                return math.inf
            sum_phi_temp += self.node_info[i]["phi"]
        return sum_phi_temp

    def h(self, n, i):
        """
        h is called the edge heuristic. It basically converts the reaction prediction evaluation into an integer.
        h assigns values to the different possible reactions i which can create the molecule n.
        The DFPN-E paper defines h(n, i) as follows: h(n, i) = min(M_pn, ⌊−log(P(n, a) + eps) + 1⌋)
        M_pn and eps are constants and P(n, a) is the probability of reaction rule a being applied to create molecule n
        predicted by a neural network. i is then the corresponding reaction if reaction rule a gets applied to n.

        Args:
            n (str): molecule in smiles format
            i (str): reaction in smiles format

        Returns:
            h_n_i (int): integer representing an evaluation of a reaction i being used on molecule n
        """
        m_pn = 20  # constant to ensure h(n, i) is not getting too big
        eps = 10 ** (-30)  # constant to ensure the term in the logarithm is not 0
        p_n_i = self.graph[n][i]["probability"]
        h_n_i = min(m_pn, math.floor(-math.log(p_n_i + eps) + 1))
        return h_n_i

    def build_tree(self, smi, tree=None, path=None, k=None, level=0):
        """
        build_tree builds the just found proof_tree and finds + saves the paths to all buyable targets

        Args:
            smi (str): molecule in smiles format
            tree (dict): dictionary of the so far build tree containing every needed molecule for the just found route
            path (list): current path that is recursively build by appending reactions and targets
            k (int): number of so far found paths to building blocks in the current route
            level (int): current recursion level of the build_tree function

        Returns:
            proof_tree (dict): dictionary of the thus far build proof_tree representing a route, at the end it should be
            a full synthesis route for the target
        """
        if tree is None:  # since this function is recursive, we need to check if smi=r (i.e. if tree is None)
            proof_tree = {}  # if smi=r the tree gets initialized
            self.mult_paths = {}  # and the dictionary of paths to building blocks needs to be reinitialized as empty
            path = []
            k = 0
        else:
            proof_tree = tree
        path.append(smi)  # path is basically the same as path_from_root, however it includes the reactions
        self.path_from_root.append(smi)
        # no_rxn_flag is a flag that indicates if the molecule smi is falsely considered for a proof, i.e. if its
        # proof number is not actually 0, that can happen if e.g. for a proven reaction of smi one of its reactants
        # closes a cycle or was itself using a falsely considered proven molecule as a reactant
        # if however we find a new reaction for the molecule smi and recursively enter the build_tree function again,
        # the no_rxn_flag needs to be set to False, because it should only be True as long as there is no actual proven
        # reaction for smi
        self.no_rxn_flag = False
        # if molecule smi is a building block, it is a leaf of the proof_tree and we can return
        if self.graph[smi] is True:
            proof_tree[smi] = {"buyable": True,
                               "reaction": None,
                               "reactants": [],
                               "depth": 0.5 * (len(path) + 1),
                               "probability": None,
                               "template": None,
                               "ff_score": None}
            self.mult_paths[k] = path[:]  # need to use a copy of path, otherwise remove will update it
            k += 1
            path.remove(smi)
            self.path_from_root.remove(smi)
            return None, k
        else:
            # check if the reaction, which used molecule smi, was proven without the proof number of smi being 0
            # if so we recompute the values of smi
            if self.node_info[smi]["phi"] != 0:
                self.recomp_mol(smi)
                # if after recomputation smi is still not proven, the reaction that used smi is falsely considered
                # proven and by setting no_rxn_flag to True, we ensure that its values get updated after returning
                if self.node_info[smi]["phi"] != 0:
                    self.no_rxn_flag = True
                    path.remove(smi)
                    self.path_from_root.remove(smi)
                    return proof_tree, k
            # rxn_found is a flag that gets set to True once a reaction rxn of molecule smi has a proof number of 0
            # if that never happens the cycle_flag gets set to True and the algorithm knows, that smi can't be proven
            # and so the values of the reaction leading to smi should be updated
            # rxn_found = False
            self.no_rxn_flag = True
            for rxn in self.graph[smi]:
                # if a reaction is proven, we don't need to look at other ones, except if the proof was false, i.e
                # no_rxn_flag is True
                if self.node_info[rxn]["delta"] == 0:
                    path.append(rxn)
                    self.rxn_from_root.append(self.graph[smi][rxn]["rxn_class_id"])
                    proof_tree[smi] = {"buyable": False,
                                       "reaction": rxn,
                                       "reactants": self.graph[rxn],
                                       "depth": 0.5 * len(path),
                                       "probability": self.graph[smi][rxn]["probability"],
                                       "template": self.graph[smi][rxn]["template"],
                                       "rxn_class": self.graph[smi][rxn]["rxn_class"],
                                       "rxn_class_id": self.graph[smi][rxn]["rxn_class_id"],
                                       "ff_score": self.graph[smi][rxn]["ff_score"]}
                    self.no_rxn_flag = False
                    cycle_flag = False
                    for reactant in proof_tree[smi]["reactants"]:
                        # if a reactant is in path_from_root, then by using rxn we would close a cycle, which can't be
                        # correct, therefore cycle_flag gets set to True and no_rxn_flag also needs to go back to True
                        if reactant in self.path_from_root:
                            cycle_flag = True
                            self.no_rxn_flag = True
                            break
                        # since the for loop goes through every reactant, if the no_rxn_flag is True for the first one,
                        # we don't want the algorithm to go through the second reactant and setting the flag to False
                        # again, so we check if reactant is maybe from a completely new reaction, because in this case
                        # the no_rxn_flag must not stay True
                        elif self.no_rxn_flag and reactant != proof_tree[smi]["reactants"][0]:
                            break
                        # rare case of a molecule being used at two different paths
                        if reactant in proof_tree:
                            continue
                        else:
                            _, k = self.build_tree(reactant, proof_tree, path, k, level=level + 1)
                    path.remove(rxn)
                    self.rxn_from_root.reverse()
                    self.rxn_from_root.remove(self.graph[smi][rxn]["rxn_class_id"])
                    self.rxn_from_root.reverse()
                    if not self.no_rxn_flag:
                        break
                    # if cycle_flag is True the last reaction rxn closes a cycle and should therefore be considered
                    # disproven for the current path
                    elif cycle_flag:
                        self.node_info[rxn].update({"phi": 0,
                                                    "delta": math.inf})
                        self.node_info[rxn]["dp_paths"].append(self.path_from_root[:])
                        unwanted_mol = proof_tree.pop(smi)
                        self.del_rest(unwanted_mol["reactants"], unwanted_mol["depth"], proof_tree)
                        continue
                    # if no_rxn_flag is True but not cycle_flag, the reaction rxn was falsely considered proven and its
                    # values get updated after first updating the values of its child nodes
                    # however it did not close a cycle, so we don't need to disprove it for the current path
                    else:
                        for reactant in self.graph[rxn]:
                            if self.graph[reactant] is True:  # skip building block
                                continue
                            self.node_info[reactant].update({"phi": self.deltaMin_OR(reactant),
                                                             "delta": self.sum_phi(reactant)})
                        self.node_info[rxn].update({"phi": self.deltaMin_AND(rxn),
                                                    "delta": self.sum_phi(rxn)})
                        unwanted_mol = proof_tree.pop(smi)
                        self.del_rest(unwanted_mol["reactants"], unwanted_mol["depth"], proof_tree)
                        continue
            # if no reaction was found for molecule smi, the no_rxn_flag stays True, because it indicates that a
            # molecule was falsely considered proven and the reaction that used it will get its values updated after the
            # "if not self.no_rxn_flag" condition
            path.remove(smi)
            self.path_from_root.remove(smi)
        return proof_tree, k

    def recomp_mol(self, smi, level=0):
        """
        recomp_mol is used to recompute the actual value of the proof- and disproof number a molecule smi, after it got
        called in the build_tree function although the proof number was unequal to 0. It recursively calls its child nodes.

        Args:
            smi (str): molecule in smiles format
            level (int): current recursion level of the recomp_mol function
        """
        level += 1  # recursion level
        # the recomputation should never go lower than max_depth recursion level
        if level < self.max_depth:
            for rxn in self.graph[smi]:
                self.recomp_rxn(rxn, level)
                # if a reaction rxn is proven we got what we wanted and we can stop the recomputation of molecule smi
                if self.node_info[rxn]["delta"] == 0:
                    break
        self.node_info[smi].update({"phi": self.deltaMin_OR(smi),
                                    "delta": self.sum_phi(smi)})
        return

    def recomp_rxn(self, rxn, level):
        """
        recomp_rxn does basically the same as recomp_mol, but it can only get called from recomp_mol

        Args:
            rxn (str): molecule in smiles format
            level (int): current recursion level of the recomp_mol function
        """
        for mol in self.graph[rxn]:
            # if rxn leads to a cycle, we need to disprove it for the current path and can stop the recomputation
            if mol in self.path_from_root:
                self.node_info[rxn].update({"phi": 0,
                                            "delta": math.inf})
                self.node_info[rxn]["dp_paths"].append(self.path_from_root[:])
                break
            # if mol is not in self.graph it has not yet been expanded and therefore it currently cannot be proven, it
            # needs to be expanded first
            elif mol not in self.graph:
                break
            # if self.graph[mol] is True mol is a building block
            elif self.graph[mol] is True:
                continue
            # otherwise we can recursively recompute the values of molecule mol
            else:
                self.path_from_root.append(mol)
                self.recomp_mol(mol, level)
                self.path_from_root.remove(mol)
                self.node_info[rxn].update({"phi": self.deltaMin_AND(rxn),
                                            "delta": self.sum_phi(rxn)})
        return

    def del_rest(self, reactants, depth, proof_tree):
        """
        del_rest deletes every molecule that has been falsely added to the proof_tree dictionary
        this is sometimes the case if the build_tree function was unable to construct a route for a second molecule that
        would be needed for a given reaction
        del_rest then recursively goes through the whole branch of a falsely added molecule

        Args:
            reactants (list): list of reactants of the reaction that build_tree was unable to prove
            depth (int): the depth of the reaction that build_tree was unable to prove
            proof_tree (dict): dictionary of the route that is currently build in the build_tree function
        """
        for reactant in reactants:
            # every reactant in reactants has been falsely added to the proof_tree, if build_tree was unable to prove
            # the whole reaction, however sometimes that is the case because of loops, in that case a molecule in
            # reactants is in the proof_tree but it would be fatal to delete ist, therefore we need to check if the
            # depth that has been saved in the proof_tree dict for the reactant is higher than the depth of the reaction
            if reactant in proof_tree and proof_tree[reactant]["depth"] > depth:
                unwanted_mol = proof_tree.pop(reactant)
                self.del_rest(unwanted_mol["reactants"], depth, proof_tree)
        return

    def updateNodeInfo(self):
        """
        updateNodeInfo recalculates the proof- and disproof numbers of targets and reactions that were used for the
        last found route, so that the algorithm can start again with an unproven target
        """
        # identify the "deepest" reaction leading to a buyable out of the just found route
        update_path = []
        for path in self.mult_paths:
            if len(update_path) <= len(self.mult_paths[path]):
                update_path = self.mult_paths[path]
        # set the reaction leading to the "deepest" buyable to disproven
        rxn = update_path[-2]
        self.node_info[rxn].update({"phi": 0,
                                    "delta": math.inf})
        # we don't want this reaction to be disproven forever, only for the just used path
        self.node_info[rxn]["dp_paths"].append(update_path[0:-2:2])
        # update (dis-)proof numbers of the impacted nodes, being every node in update_path[0:-2]
        for i, node in enumerate(reversed(update_path[0:-2])):
            if i % 2 == 0:  # if i is a molecule
                self.node_info[node].update({"phi": self.deltaMin_OR(node, True),
                                             "delta": self.sum_phi(node)})
            else:  # if i is a reaction
                self.node_info[node].update({"phi": self.deltaMin_AND(node),
                                             "delta": self.sum_phi(node)})
                # the penalty should only be added to reactions that are not still proven
                # this sometimes happens if a molecule has a bunch of reactions that all lead to different building blocks
                # without this if clause we would always only consider the first found last reaction, although other
                # ones could potentially be much better
                # the algorithm then basically instantly gives out every one of those routes
                if self.node_info[node]["delta"] != 0:
                    # penalty should be in the range of 0 - 20, 5 - 10
                    # tune the penalty that should be added to every used reaction
                    self.node_info[node]["delta"] += self.mult_paths_penalty
        self.node_info[rxn].update({"phi": 1,
                                    "delta": len(self.graph[rxn])})
