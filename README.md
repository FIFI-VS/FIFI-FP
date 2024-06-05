# FIFI

FIFI is a molecular fingerprint, containing hybrid information from ligand- and structure-based drug design.

#Requirements
Python 3
NumPy
Pandas
RDKit

#How to use
- Prepare the files containing sdf of docked ligands, and pdb of the receptor
- Use fifi vicinity module to extract vicinity atoms
- Use fifi generating fp module to translate the atoms into fingerprints, saved as pickle
- Use fifi pickle load module to get a dataframe version from the generated FIFI FP, and further can be used as a machine learning input
