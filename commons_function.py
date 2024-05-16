from rdkit import Chem

import matplotlib as plt

def sarm_lister(target_folder, target_abv):
    target = dictionary_target[target_abv]
    sarm_list = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
    
    return sarm_list
    
def dictionary_target(target_abv):
    target_dict = {"kor": "Kappa_opioid_receptor", 
                   "mapk": "MAP_kinase_ERK2", 
                   "adrb2": "Beta-2_adrenergic_receptor", 
                   'casp1': 'Caspase-1'
                  }
    
    target = target_dict[target_abv]

    return target


def dictionary_model(model_abv):
    model_dict = {'Random Forest': 'rf', 
                  'Linear Kernel SVM': 'lksvm', 
                  'Radial Based Function SVM': 'rbfsvm',
                  "Logistic Regression": 'logreg', 
                  "Tanimoto Kernel SVM": 'tksvm'}
    model_abv_dict = {v: k for k, v in model_dict.items()}

    model = model_abv_dict[model_abv]

    return model


def df_dollar2underscore(dataframe_dollar):
    updated_dataframe = dataframe_dollar.applymap(lambda x: str(x).replace('$', '_'))
    updated_dataframe.index = updated_dataframe.index.str.replace('$', '_')

    return updated_dataframe


def common_series_compiller(series_1, series_2):
    common_indexes = series_2.index.intersection(series_1.index)

    series_2 = series_2.loc[common_indexes]
    series_1 = series_1.loc[common_indexes]

    return series_1, series_2


def set_font_image():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']


def get_mols_from_stream(suppl):
    fail_mols = 0
    mols = []
    for x in suppl:
        if x is None:
            fail_mols += 1
            continue
        mols.append(x)
    return mols, fail_mols


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
