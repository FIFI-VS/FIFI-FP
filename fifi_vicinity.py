import os
import pickle
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Union

from rdkit import Chem


def parse_atom_line_pdb(line):
    if 'HETATM' == line[:6]:
        at = 'HETATM'
    elif 'ATOM' == line[:4]:
        at = 'ATOM'
    else:
        return None

    res_name = line[17:20]
    chain_identifer = line[21]
    seq_number = line[22:26]
    return at, res_name, chain_identifer, seq_number


def get_mols_from_stream(suppl):
    fail_mols = 0
    mols = []
    for x in suppl:
        if x is None:
            fail_mols += 1
            continue
        mols.append(x)
    return mols, fail_mols


def load_pdb_file_with_amino_acid(fname: str,
                                  sanitize: bool = False,
                                  remove_hetatoms: bool = True):
    mol = Chem.MolFromPDBFile(fname, sanitize=sanitize, removeHs=False)
    atypes = []
    with open(fname, 'r') as rfp:
        line = rfp.readline()
        while line:
            parsed_cmps = parse_atom_line_pdb(line)
            if parsed_cmps is not None:
                at, rn, ci, sn = parsed_cmps
                atypes.append((at, rn, ci, sn))
            line = rfp.readline()

    natoms = mol.GetNumAtoms()
    if natoms != len(atypes):
        raise AssertionError('The number of atoms between mol and file is different')

    #set residue inf as atom property
    propnames = 'atomtype residue_name chain_identifer sequence_number'.split()
    for atom, cmps in zip(mol.GetAtoms(), atypes):
        for cmp, pname in zip(cmps, propnames):
            atom.SetProp(pname, cmp)

    if remove_hetatoms:
        rwmol = Chem.EditableMol(mol)
        HETATOMs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetProp(propnames[0]) == 'HETATM']
        for hetatom in reversed(HETATOMs):
            rwmol.RemoveAtom(hetatom)
        mol = rwmol.GetMol()

    return mol


def dict2pickle(dict_name,
                pickle_path_name):
    with open(pickle_path_name, "wb") as handle:
        pickle.dump(dict_name, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_mols_from_sdf(fname: str,
                       return_counts: bool = False,
                       remove_Hs: bool = True
                       ):
    suppl = Chem.SDMolSupplier(fname, removeHs=remove_Hs)
    mols, fail_mols = get_mols_from_stream(suppl)

    if return_counts:
        return mols, fail_mols
    else:
        return mols


def boolean_lister(list_or_not):
    if isinstance(list_or_not, list):
        list_extracted = list_or_not
    else:
        list_extracted = [list_or_not]

    return list_extracted


def extract_close_atoms(protein_pdb: Chem.Mol,
                        ligands_sdf: list,
                        thres: Union[list, float] = 5.5):
    """
    extracting close atom
    """

    thres = boolean_lister(thres)

    nc_closeatoms = []
    nligands = len(ligands_sdf)

    for idx, ligand in enumerate(ligands_sdf):
        print(f'processing {idx + 1}/{nligands}', end='\r')

        natomlig = ligand.GetNumAtoms()
        catmol = Chem.CombineMols(ligand, protein_pdb)
        distmat = Chem.Get3DDistanceMatrix(catmol)
        nc_thrs_results = dict()

        for thr in thres:
            okdmat = (distmat <= thr)
            protok = np.where(okdmat[:natomlig, natomlig:])
            nc_ligpro_atommap = defaultdict(list)

            for ligaidx, proaidx in zip(*protok):
                res_inf = (protein_pdb.GetAtomWithIdx(int(proaidx))).GetPropsAsDict()

                nc_ligpro_atommap[ligaidx].append((proaidx, res_inf))

            nc_thrs_results[thr] = nc_ligpro_atommap

        nc_closeatoms.append(nc_thrs_results)

    return nc_closeatoms


def ligand_interaction_comprehension(close_atom_list,
                                     ligands_sdf,
                                     thres_list: Union[list, float] = 5.5):
    frag_inter_list = []

    thres_list = boolean_lister(thres_list)

    for idx, ligand in enumerate(ligands_sdf):
        ligand_name = ligand.GetProp('_Name')

        for distance_threshold in thres_list:
            for i, interacting_atom in enumerate(close_atom_list[idx][distance_threshold].copy()):
                amino_boolean_checker = close_atom_list[idx][distance_threshold][i]

                unique_set_number = set()

                for _, tup in amino_boolean_checker:
                    seq_number = tup.get("sequence_number")

                    if seq_number is not None:
                        unique_set_number.add(seq_number)

                unique_set_number_list = list(unique_set_number)
                unique_set_number = [x for x in unique_set_number_list if x != "NaN"]
                unique_set_number = [int(x) for x in unique_set_number]

                fragment_interaction_tuple = (idx, ligand_name, distance_threshold, interacting_atom, unique_set_number)

                frag_inter_list.append(fragment_interaction_tuple)

    df_fragment_interaction = pd.DataFrame(frag_inter_list, columns=['idx', 'name', 'threshold', 'atom_id', 'aa'])

    return df_fragment_interaction


