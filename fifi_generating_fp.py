import csv
import os
import pickle

import numpy as np
import pandas as pd
from commons_function import load_mols_from_sdf
from scripts_module import get_morgan_features, make_ecfp_substruct_from_hash


def residue_dict_maker(pdb_file):
    amino_acids_dict = {}

    with open(pdb_file, 'r') as pdb:
        for line in pdb:
            if line.startswith('ATOM'):
                residue_order = int(line[22:26].strip())
                amino_acid_code = line[17:20].strip()
                if amino_acid_code != 'HOH':
                    amino_acids_dict[residue_order] = amino_acid_code

    return amino_acids_dict


def generating_hash_dictionary(ligands_mol, ecfp_dist):
    radii = int(ecfp_dist / 2)
    dict_smarts = {}
    dict_hash = {}

    for mol in ligands_mol:
        dict_key = mol.GetProp("_Name")

        atomwise_hash = get_morgan_features(mol,
                                            radius=radii,
                                            return_as_string=False,
                                            input_smiles=False,
                                            return_atom_hash=True,
                                            include_redundant_environment=True)

        hashkeys = list(atomwise_hash.values())
        hashkeys = [v for arr in hashkeys for v in arr]
        uhashkeys = np.unique(hashkeys)

        smarts = make_ecfp_substruct_from_hash([mol],
                                               uhashkeys,
                                               radii,
                                               inc_redundant_environment=True,
                                               return_recursive_smarts=True)

        dict_smarts.update(smarts)
        dict_hash[dict_key] = atomwise_hash

    return dict_smarts, dict_hash


def separating_amino_acid_list(threshold_df,
                               aa_column_name="aa"):
    flattened_series = threshold_df[aa_column_name].explode()
    unique_numbers = flattened_series.unique()
    unique_numbers_cleaned = [x for x in unique_numbers if str(x).isdigit()]
    unique_numbers_sorted = sorted(unique_numbers_cleaned)

    return unique_numbers_sorted


def returning_aa_with_num(aa_name, residue_dict):
    try:
        aa_name = "".join([x for x in aa_name if x.isdigit()])
    except TypeError:
        pass

    aa_name = int(aa_name)
    value = residue_dict[aa_name]

    return f"{aa_name}{value}"


def traverse_neighbor(mol, atom_index, num_bonds):
    atom = mol.GetAtomWithIdx(atom_index)
    neighbors = set()
    visited = set()

    def traverse(atom_el, distance):
        visited.add(atom_el.GetIdx())

        if distance == 0:
            return

        for neighbor in atom_el.GetNeighbors():
            if neighbor.GetAtomicNum() == 1: #skipping hydrogen, if any
                continue

            if neighbor.GetIdx() not in visited:
                neighbors.add(neighbor.GetIdx())

                traverse(neighbor, distance - 1)

    traverse(atom, num_bonds)

    return list(neighbors)


def dod_extraction(dod, list_key):
    unique_values = []

    for key in list_key:
        inner_dict = dod.get(key, {})

        if isinstance(inner_dict, list):
            unique_values.extend(inner_dict)

        else:
            unique_values.extend(inner_dict.values())

    unique_values_set = set(unique_values)

    return unique_values_set


def updating_hash_dictionary(dictionary_to_update, key_dictionary, hash_array):
    if key_dictionary in dictionary_to_update:
        dictionary_to_update[key_dictionary] = list(set(dictionary_to_update[key_dictionary] + hash_array))

    else:
        dictionary_to_update[key_dictionary] = hash_array

    return dictionary_to_update


def dict_order2smarts(smarts_dict,
                      order_dict
                      ):
    for old_key_b, new_key_b in smarts_dict.items():
        if old_key_b in order_dict:
            value = order_dict.pop(old_key_b)
            order_dict[new_key_b] = value

    return order_dict


def saving_whole_dict(whole_unique_dict,
                      path_to_save
                      ):
    ordered_whole_dict = {}
    for key, inner_dict in whole_unique_dict.items():
        sorted_items = sorted(inner_dict.items(), key=lambda x: x[1])
        ordered_dict = dict(sorted_items)
        ordered_whole_dict[key] = ordered_dict

    with open(path_to_save, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for key1, value1 in ordered_whole_dict.items():
            for key2, value2 in value1.items():
                writer.writerow([key1, key2, value2])


def generating_fifi_fp(ligand_mols,
                       df_boolean_frag,
                       lig_depth,
                       residue_dict,
                       ecfp_dist=2,
                       threshold_or_list=5.5,
                       smarts_save_dir="{DATADIRPATH}/fifi/smarts_bits_v8/{target_abv}/",
                       number_ecfp_bits=1024
                       ):
    dict_smarts, dict_hash = generating_hash_dictionary(ligand_mols, ecfp_dist)

    list_ligands_name = df_boolean_frag["name"].unique()
    lig_num = len(list_ligands_name)

    if isinstance(threshold_or_list, list):
        threshold_list = threshold_or_list
    else:
        threshold_list = [threshold_or_list]

    for threshold in threshold_list:
        threshold_save_dir = f"{smarts_save_dir}/{threshold}"
        pickle_save_dir = f"{threshold_save_dir}/ecfp{ecfp_dist}_n{lig_depth}"

        os.makedirs(pickle_save_dir, exist_ok=True)

        threshold_df = df_boolean_frag[df_boolean_frag['threshold'] == threshold]

        aa_list = separating_amino_acid_list(threshold_df)

        hash_aa_dict = {}
        whole_hash_dict, whole_unique_dict = {}, {}

        for aa_order, aa_name in enumerate(aa_list):
            filtered_df = threshold_df[threshold_df["aa"].apply(lambda lst: aa_name in lst)]
            if filtered_df.empty:
                continue

            residue_name_order = returning_aa_with_num(aa_name, residue_dict)

            order_dict, padded_order_dict = {}, {}

            ligand_smarts_list = []

            for idx, mol in enumerate(ligand_mols):
                print(
                    f'processing {residue_name_order} {aa_order + 1}/{len(aa_list)} aa {idx}/{lig_num} ligands',
                    end='\r')
                ligand_name = mol.GetProp('_Name')

                name_df = filtered_df[filtered_df['name'] == ligand_name]

                if len(name_df) == 0:
                    continue

                neighboring_atom_indices_set = set()
                filtered_atoms = name_df["atom_id"].unique().tolist()

                if lig_depth == 0:
                    atom_and_neighbours = sorted(filtered_atoms)

                else:
                    for atom_index in filtered_atoms:
                        try:
                            neighboring_atom_indices = traverse_neighbor(mol, atom_index, lig_depth)
                            neighboring_atom_indices_set.update(neighboring_atom_indices)
                        except TypeError:
                            continue

                    atom_and_neighbours = list(set(filtered_atoms) | neighboring_atom_indices_set)
                    atom_and_neighbours = sorted(atom_and_neighbours)

                unique_hashes_cpd = dod_extraction(dict_hash[ligand_name], atom_and_neighbours)

                bit_array_cpd = [x % number_ecfp_bits + (number_ecfp_bits * aa_order) for x in unique_hashes_cpd]

                for value in unique_hashes_cpd:
                    if value not in ligand_smarts_list:
                        ligand_smarts_list.append(value)

                order_dict.update({item: order for order, item in enumerate(ligand_smarts_list)})

                padded_order_dict[ligand_name] = list(set(unique_hashes_cpd))

                hash_aa_dict = updating_hash_dictionary(hash_aa_dict, ligand_name, bit_array_cpd)  #this creates FIFI BA

            for key, value_list in padded_order_dict.items():  #changing hash to order
                padded_order_dict[key] = [order_dict.get(item, item) for item in value_list]

            #dumping the pickle for each AA
            padded_order_dict = {x: sorted(padded_order_dict[x]) for x in padded_order_dict.keys()}

            with open(f"{pickle_save_dir}/{aa_name}.pickle", "wb") as handle:
                pickle.dump(padded_order_dict, handle)

            hash_dict = order_dict.copy()
            order_dict = dict_order2smarts(dict_smarts, order_dict)

            whole_unique_dict.update({residue_name_order: order_dict})
            whole_hash_dict.update({residue_name_order: hash_dict})

        whole_unique_dict_filename = f"{smarts_save_dir}/dict_unique_aa_fifi_ecfp{ecfp_dist}_n{lig_depth}.csv"
        whole_hash_dict_filename = f"{smarts_save_dir}/hashdict_unique_aa_fifi_ecfp{ecfp_dist}_n{lig_depth}.csv"

        saving_whole_dict(whole_unique_dict, whole_unique_dict_filename)
        saving_whole_dict(whole_hash_dict, whole_hash_dict_filename)

        with open(f"{smarts_save_dir}/ecfp{ecfp_dist}_n{lig_depth}_bits.pickle", "wb") as handle:
            pickle.dump(hash_aa_dict, handle)


def generating_fifius_only(ligand_mols,
                           df_boolean_frag,
                           threshold_or_list,
                           lig_depth,
                           res_dict,
                           ecfp_dist,
                           smarts_save_dir="{DATADIRPATH}/fifi/smarts_bits_v8/{target_abv}/",
                           ):
    dict_smarts, dict_hash = generating_hash_dictionary(ligand_mols, ecfp_dist)

    list_ligands_name = df_boolean_frag["name"].unique()
    lig_num = len(list_ligands_name)

    if isinstance(threshold_or_list, list):
        threshold_list = threshold_or_list
    else:
        threshold_list = [threshold_or_list]

    for threshold in threshold_list:
        threshold_save_dir = f"{smarts_save_dir}/{threshold}"
        pickle_save_dir = f"{threshold_save_dir}/ecfp{ecfp_dist}_n{lig_depth}"

        os.makedirs(pickle_save_dir, exist_ok=True)

        threshold_df = df_boolean_frag[df_boolean_frag['threshold'] == threshold]

        aa_list = separating_amino_acid_list(threshold_df)

        whole_hash_dict, whole_unique_dict = {}, {}

        for aa_order, aa_name in enumerate(aa_list):
            filtered_df = threshold_df[threshold_df["aa"].apply(lambda lst: aa_name in lst)]
            if filtered_df.empty:
                continue

            residue_name_order = returning_aa_with_num(aa_name, res_dict)

            order_dict, padded_order_dict = {}, {}

            ligand_smarts_list = []

            for idx, mol in enumerate(ligand_mols):
                print(
                    f'processing {residue_name_order} {aa_order + 1}/{len(aa_list)} aa {idx}/{lig_num} ligands',
                    end='\r')
                ligand_name = mol.GetProp('_Name')

                name_df = filtered_df[filtered_df['name'] == ligand_name]

                if len(name_df) == 0:
                    continue

                neighboring_atom_indices_set = set()
                filtered_atoms = name_df["atom_id"].unique().tolist()

                if lig_depth == 0:
                    atom_and_neighbours = sorted(filtered_atoms)

                else:
                    for atom_index in filtered_atoms:
                        try:
                            neighboring_atom_indices = traverse_neighbor(mol, atom_index, lig_depth)
                            neighboring_atom_indices_set.update(neighboring_atom_indices)
                        except TypeError:
                            continue

                    atom_and_neighbours = list(set(filtered_atoms) | neighboring_atom_indices_set)
                    atom_and_neighbours = sorted(atom_and_neighbours)

                unique_hashes_cpd = dod_extraction(dict_hash[ligand_name], atom_and_neighbours)

                for value in unique_hashes_cpd:
                    if value not in ligand_smarts_list:
                        ligand_smarts_list.append(value)

                order_dict.update({item: order for order, item in enumerate(ligand_smarts_list)})

                padded_order_dict[ligand_name] = list(set(unique_hashes_cpd))

            for key, value_list in padded_order_dict.items():
                padded_order_dict[key] = [order_dict.get(item, item) for item in value_list]

            padded_order_dict = {x: sorted(padded_order_dict[x]) for x in padded_order_dict.keys()}

            with open(f"{pickle_save_dir}/{aa_name}.pickle", "wb") as handle:
                pickle.dump(padded_order_dict, handle)

            hash_dict = order_dict.copy()
            order_dict = dict_order2smarts(dict_smarts, order_dict)

            whole_unique_dict.update({residue_name_order: order_dict})
            whole_hash_dict.update({residue_name_order: hash_dict})

        whole_unique_dict_filename = f"{smarts_save_dir}/dict_unique_aa_fifi_ecfp{ecfp_dist}_n{lig_depth}.csv"
        whole_hash_dict_filename = f"{smarts_save_dir}/hashdict_unique_aa_fifi_ecfp{ecfp_dist}_n{lig_depth}.csv"

        saving_whole_dict(whole_unique_dict, whole_unique_dict_filename)
        saving_whole_dict(whole_hash_dict, whole_hash_dict_filename)


def generating_fifiba_only(lig_mols,
                           df_boolean_frag,
                           threshold_or_list,
                           lig_depth,
                           residue_dict,
                           ecfp_dist,
                           smarts_save_dir="{DATADIRPATH}/fifi/smarts_bits_v8/{target_abv}/",
                           number_ecfp_bits=1024
                           ):
    _, dict_hash = generating_hash_dictionary(lig_mols, ecfp_dist)

    list_ligands_name = df_boolean_frag["name"].unique()
    lig_num = len(list_ligands_name)

    if isinstance(threshold_or_list, list):
        threshold_list = threshold_or_list
    else:
        threshold_list = [threshold_or_list]

    for threshold in threshold_list:
        threshold_save_dir = f"{smarts_save_dir}/{threshold}"

        os.makedirs(threshold_save_dir, exist_ok=True)

        threshold_df = df_boolean_frag[df_boolean_frag['threshold'] == threshold]

        aa_list = separating_amino_acid_list(threshold_df)

        hash_aa_dict = {}

        for aa_order, aa_name in enumerate(aa_list):
            filtered_df = threshold_df[threshold_df["aa"].apply(lambda lst: aa_name in lst)]
            if filtered_df.empty:
                continue

            residue_name_order = returning_aa_with_num(aa_name, residue_dict)

            order_dict, padded_order_dict = {}, {}

            ligand_smarts_list = []

            for idx, mol in enumerate(lig_mols):
                print(
                    f'processing {residue_name_order} {aa_order + 1}/{len(aa_list)} aa {idx}/{lig_num} ligands',
                    end='\r')
                ligand_name = mol.GetProp('_Name')

                name_df = filtered_df[filtered_df['name'] == ligand_name]

                if len(name_df) == 0:
                    continue

                neighboring_atom_indices_set = set()
                filtered_atoms = name_df["atom_id"].unique().tolist()

                if lig_depth == 0:
                    atom_and_neighbours = sorted(filtered_atoms)

                else:
                    for atom_index in filtered_atoms:
                        try:
                            neighboring_atom_indices = traverse_neighbor(mol, atom_index, lig_depth)
                            neighboring_atom_indices_set.update(neighboring_atom_indices)
                        except TypeError:
                            continue

                    atom_and_neighbours = list(set(filtered_atoms) | neighboring_atom_indices_set)
                    atom_and_neighbours = sorted(atom_and_neighbours)

                unique_hashes_cpd = dod_extraction(dict_hash[ligand_name], atom_and_neighbours)

                bit_array_cpd = [x % number_ecfp_bits + (number_ecfp_bits * aa_order) for x in unique_hashes_cpd]

                order_dict.update({item: order for order, item in enumerate(ligand_smarts_list)})

                padded_order_dict[ligand_name] = list(set(unique_hashes_cpd))

                hash_aa_dict = updating_hash_dictionary(hash_aa_dict, ligand_name, bit_array_cpd)

        with open(f"{smarts_save_dir}/ecfp{ecfp_dist}_n{lig_depth}_bits.pickle", "wb") as handle:
            pickle.dump(hash_aa_dict, handle)
