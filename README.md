# FIFI (Fragmented Interaction FIngerprint)

FIFI is a molecular fingerprint, containing hybrid information from ligand- and structure-based drug design.

**Requirements**
- Python 3 (FIFI was initially developed in Python 3.8)
- NumPy
- Pandas
- RDKit

**How to use**
- Prepare the files containing sdf of docked ligands, and pdb of the receptor
- Use fifi_vicinity module to extract vicinity atoms
- Use fifi_generating_fp module to translate the atoms into fingerprints, saved as pickle
- Use fifi_pickle_load module to get a dataframe version from the generated FIFI FP, and further can be used as a machine learning input

**Example to use**

please refer to the jupyter notebook [FIFI example](https://github.com/FIFI-VS/FIFI-FP/blob/main/FIFI_example.ipynb)
