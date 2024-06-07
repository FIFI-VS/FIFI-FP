from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem

import ast
import re
import numpy as np


def organize_morgan_hash_on_atoms(mol: Chem.Mol, bitinf: dict, organize_radius: bool = False):
    if organize_radius:
        ret_dict = defaultdict(dict)
    else:
        ret_dict = defaultdict(str)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        hash_radi = []
        for hashval, val in bitinf.items():
            ok_hash = [(hashval, v[1]) for v in val if v[0] == idx]
            hash_radi += ok_hash
        ordered = sorted(hash_radi, key=lambda x: x[1])
        if organize_radius:
            dict_hashes = dict()
            for v in ordered:
                dict_hashes[v[1]] = v[0]
            ret_dict[idx] = dict_hashes
        else:
            ret_dict[idx] = [v[0] for v in ordered]  # only select hash value
    return ret_dict


def get_morgan_features(orgmol, 
                        radius=2, 
                        return_as_string=False, 
                        input_smiles=False, 
                        return_atom_hash=False, 
                        include_redundant_environment=False,
                        return_bitinf=False,
                        organize_radius=False):
    """
    include_redundant_environment must be True when you want to generate atom-based fingerprints.
    :param inc_redundant_environment: must be important
    :param radi: radius for the morgan fingerprint

    """
    if input_smiles:
        try:
            mol = Chem.MolFromSmiles(orgmol)
        except:
            print("Smiles to mol conversion fail.")
            return None
    else:
        mol = orgmol

    if mol is None:
        print('Molecule is None. Return empty data')
        return None

    binf = dict()
    features = AllChem.GetMorganFingerprint(mol, radius=radius, bitInfo=binf, includeRedundantEnvironments=include_redundant_environment)
    finf     = features.GetNonzeroElements()
    
    # check the unnecessary hash or not
    newbinf = dict()
    if include_redundant_environment: # True only chance of generating unmatched hashes
        for hkey, envs in binf.items():
            rinf = list()
            for env in envs:
                atom_idx = env[0]
                radi     = env[1]
                env      = AllChem.FindAtomEnvironmentOfRadiusN(mol, radi, atom_idx)
                # the hash does not corresponds to environment
                if (radi != 0) and (len(env) == 0):
                    print('unnecessary hashes are detected. remove the hash from the mol', hkey, 'from', Chem.MolToSmiles(mol))
                    trancated =True
                    continue
                rinf.append((atom_idx, radi))
            newbinf[hkey] = tuple(rinf)
        binf = newbinf
    
    # return options
    if return_atom_hash:
        atomidx_hash = organize_morgan_hash_on_atoms(mol, binf, organize_radius)
        if return_bitinf:
            return dict(atomhash=atomidx_hash, bitinf=binf)
        else:
            return atomidx_hash

    if return_as_string:
        return str(finf)
    else:
        return finf


def make_ecfp_substruct_from_hash(mols, 
                                  hkeys,
                                  radi, 
                                  include_redundant_environment = True,
                                  return_recursive_smarts=False, 
                                  check_exhastives=False,
                                  return_radius_dict=False):
    """
    transforming the hash keys into smiles format
    :param mols:
    :param hkeys:
    :param radi: radius for the morgan fingerprint
    include_redundant_environment: must be important
    v3: must be most accurate based on the rdkit definition.
    :return:
    """
    nhash = len(hkeys)
    hash_found = dict.fromkeys(hkeys, 0)  # flag (the hash key is found)
    substructs = dict.fromkeys(hkeys) # found substructures corresponding to the hash
    radiuss    = dict.fromkeys(hkeys) # found substructures corresponding to the hash

    for mol in mols:
        sub_infos={}
        fp = AllChem.GetMorganFingerprint(mol, radi, bitInfo=sub_infos, includeRedundantEnvironments=include_redundant_environment)

        for hkey, struct_inf in sub_infos.items():
            if hkey not in hash_found.keys():
                continue

            if hash_found[hkey]:  # the hash is already searched
                continue

            hash_found[hkey] = True
            # substructure extraction options
            smarts           = make_canon_smarts_from_sub_v3(mol, struct_inf, return_recursive_smarts, include_ring=True) 
            substructs[hkey] = smarts
            radiuss[hkey]    = struct_inf[0][1] # extract representative

            
            # the query find all the atoms registed
            if check_exhastives: 
                # debugging
                if smarts == "":  # the hash must be eliminated
                    print('hash', hkey, 'does not match any substructure')
                    continue
                
                query    = substructs[hkey]
                sub_sets = mol.GetSubstructMatches(AllChem.MolFromSmarts(query))
                atomsets = [v[0] for v in struct_inf]
                passtest = True
                # check the found substructure sizes
                if len(sub_sets) != len(atomsets):
                    passtest = False
                
                # check whether the registered atoms are found in substs
                found_sub = np.zeros(len(sub_sets), dtype=bool)
                for atom in atomsets:
                    found_mask = [atom in subset for subset in sub_sets]
                    found_sub[found_mask] = 1
                if np.sum(found_sub) != len(sub_sets):
                    passtest = False
                
                if not passtest:
                    # for debugging
                    print('inconsistency finding, check the query:', query, 'mol', AllChem.MolToSmiles(mol), 'registerd_atomsets', atomsets)
                    make_canon_smarts_from_sub_v3(mol, struct_inf, return_recursive_smarts, include_ring=True) 
                
        if sum(hash_found.values()) == nhash:
            if return_radius_dict:
                return substructs, radiuss
            else:
                return substructs

    return ["ERROR all the hash cannot be detected: only " + str(sum(hash_found)) + " hashes were detected"]


def make_canon_smarts_from_sub_v3(amol, ecfp_inf, return_recursive_smarts=False, include_ring=True):
    """
    make the canonical smiles corresponding to the ecfp_inf substructure
    inputs:
    ---------------------
    amol: molecule object
    ecfp_inf: hash information obtained from GetMorganFingerprint function by RDKit
    :return:
    Chem.MolToSmarts cannot work for generating the atom centered SMARTS...
    """

    atom_idx = ecfp_inf[0][0]
    radi = ecfp_inf[0][1]
    return make_atombased_smarts(amol, atom_idx, radi, return_recursive_smarts, include_ring)


def make_atombased_smarts(mol, atom_idx, radius, return_recursive_smarts, include_ring=True):
    env = AllChem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    amap = {}
    submol = AllChem.PathToSubmol(mol, env, atomMap=amap)
    #natoms = submol.GetNumAtoms()

    if len(env) == 0 and radius != 0:
        #No such substructure found
        return ""

    #Modified SMARTS
    atomquery = r'\[.*?\]'  # *? is non-greedy fashion

    if radius == 0:  #
        catom = mol.GetAtomWithIdx(atom_idx)
        word = make_atom_smarts(catom, include_ring)
        smarts = word
    else:
        rev_amap = {v: k for k, v in amap.items()}  # sub -> parent atoms
        smarts = AllChem.MolToSmiles(submol, rootedAtAtom=amap[atom_idx], allHsExplicit=True, allBondsExplicit=True)
        atom_order = ast.literal_eval(submol.GetProp('_smilesAtomOutputOrder'))

        for idx, aidx in enumerate(atom_order):
            atom = submol.GetAtomWithIdx(aidx)
            parent_atom = mol.GetAtomWithIdx(rev_amap[aidx])  # to correct degree
            word = make_atom_smarts(atom, include_ring, parent_atom)
            smarts = replace_nth(smarts, atomquery, word, n=idx + 1)

    if return_recursive_smarts:
        return '[$(' + smarts + ')]'

    return smarts


def replace_nth(string, sub, rep, n):
    # utilty func only replace n-th occurence of the findings
    where = [m.start() for m in re.finditer(sub, string)][n - 1]  # previous one
    before = string[:where]
    after = string[where:]
    mafter = re.sub(sub, rep, after, 1)  # only the first occurence is changed
    if after == mafter:
        raise ValueError('no option for replacement. Check SMILES.')
    new_string = before + mafter
    return new_string


def make_atom_smarts(atom, include_ring, parent_atom=None):
    if parent_atom is None:
        parent_atom = atom

    deg = parent_atom.GetTotalDegree()
    n_h = atom.GetTotalNumHs()
    anum = atom.GetAtomicNum()
    chg = '{:+d}'.format(atom.GetFormalCharge())
    # default setting RDKit this is off
    if include_ring:
        ring = 'R' if parent_atom.IsInRing() else 'R0'  # ring should be optional but default is true
    else:
        ring = ''
    word = '[#{a}X{d}{c}H{h}{R}]'.format(a=anum, d=deg, c=chg, h=n_h, R=ring)
    return word
