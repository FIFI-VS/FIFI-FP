import os
import pickle
import numpy as np
import pandas as pd


def extract_number(item):
    return int(item.split(".")[0])


def filename2aaname(filename):
    aa_name = filename.split('.')[0]
    return aa_name


def array2df(numpy_array, 
             list_ligand_names, 
             aa_name):
    bit_list = [f'{aa_name}_{number}' for number in range(len(numpy_array))]

    return pd.DataFrame(numpy_array.T, columns=bit_list, index=list_ligand_names).astype(int)


def zero_aadf(num_bits, 
              list_ligand_names, 
              aa_name):
    bit_list = [f'{aa_name}_{number}' for number in range(num_bits)]
    data = [[0] * num_bits] * len(list_ligand_names)

    return pd.DataFrame(data, columns=bit_list, index=list_ligand_names)


def list_dict2array_dict(dict_one_based, 
                         num_rows, 
                         compound_names):
    result_dict = {}

    for compound_name in compound_names:
        item = dict_one_based.get(compound_name, [])
        result_dict[compound_name] = item

    column_arrays = {key: np.zeros(num_rows) for key in result_dict.keys()}

    for col_name, indices in result_dict.items():
        column_arrays[col_name][indices] = 1

    data_array = np.column_stack(list(column_arrays.values()))

    return data_array


def partial_fifi_unique_pickle2df(pickle_folder_path, 
                                  aa_name, 
                                  list_included_ligands):
    with open(f"{pickle_folder_path}/{aa_name}.pickle", "rb") as handle:
        row_order_dict = pickle.load(handle)

    max_item = max(max(lst) for lst in row_order_dict.values()) + 1

    try:
        filtered_dict = {key: row_order_dict[key] for key in list_included_ligands if key in row_order_dict}

        array_dict = list_dict2array_dict(filtered_dict, max_item, list_included_ligands)

        fifi_aa_df = array2df(array_dict, list_included_ligands, aa_name)

    except KeyError:
        fifi_aa_df = zero_aadf(max_item, list_included_ligands, aa_name)

    return fifi_aa_df


def fifiba_pickle2df(fifi_ba_folder_path,
                     aa_list,
                     index_list,
                     ecfp_bits_number=1024):
    #fifi_ba_folder_path: path to a folder where the pickle of FIFI-BA was saved
    #included_ligands: list of compound names to be loaded from the pickle

    cpd_hash_list = pd.read_pickle(fifi_ba_folder_path)

    new_column_names = []
    for a in aa_list:
        for b in range(0, ecfp_bits_number):
            new_column_name = f'{a}_{b}'
            new_column_names.append(new_column_name)

    num_rows = len(index_list)
    num_cols = len(new_column_names)

    ecfp_df = np.zeros((num_rows, num_cols), dtype=int)

    for i, index_name in enumerate(index_list):
        columns_to_change = cpd_hash_list.get(index_name, [])

        if columns_to_change:
            ecfp_df[i, columns_to_change] = 1

    ecfp_df_merged = pd.DataFrame(data=ecfp_df, index=index_list, columns=new_column_names)
    return ecfp_df_merged


def fifius_pickle2df(fifius_folder_path,
                     included_ligands):
    #fifius_folder_path: path to a folder where all residue pickles of FIFI-US were saved
    #included_ligands: list of compound names to be loaded from the pickle
    pickle_files = [f for f in os.listdir(fifius_folder_path) if f.endswith(".pickle")]
    pickle_files = sorted(pickle_files, key=extract_number)
    fifi_whole_df = pd.DataFrame()

    for pickle_file in pickle_files:
        aa_name = filename2aaname(pickle_file)

        fifi_aa_df = partial_fifi_unique_pickle2df(fifius_folder_path, aa_name, included_ligands)

        if fifi_aa_df is not None:
            fifi_whole_df = pd.concat([fifi_whole_df, fifi_aa_df], axis=1)

    return fifi_whole_df
