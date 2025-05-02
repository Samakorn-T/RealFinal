#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 02:15:47 2024

@author: padprow
"""

import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw

def get_compound_info(smiles):
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    
    # Check if molecule conversion was successful
    if mol is None:
        return "Invalid SMILES", None

    # Get PubChem CID based on SMILES
    compounds = pcp.get_compounds(smiles, 'smiles')
    if compounds:
        compound_name = compounds[0].iupac_name  # Retrieve IUPAC name
    else:
        compound_name = "Name not found in PubChem"

    # Draw the molecule
    img = Draw.MolToImage(mol, size=(300, 300))
    
    return compound_name, img

# Input SMILES string
smiles_input = "Oc1ccccc1"
name, image = get_compound_info(smiles_input)

# Display name and image
print(f"Compound Name: {name}")
display(image)