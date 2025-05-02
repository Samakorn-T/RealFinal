#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:52:06 2024

@author: padprow
"""

import time
import pandas as pd
from tqdm import tqdm  # For progress tracking
import pubchempy as pcp
from requests.exceptions import RequestException, HTTPError
import os

# Function to extract CAS numbers from PubChemPy data using CID
def get_cas_from_cid(cid=None):
    """Retrieves CAS numbers from CID using PubChemPy.

    Args:
        cid (int, optional): The CID of the compound.

    Returns:
        list: A list of CAS numbers associated with the compound, or None if no CAS numbers are found.
    """
    cas_numbers = []

    # Attempt to retrieve CAS numbers from CID
    if cid:
        try:
            compound_cid = pcp.Compound.from_cid(cid)
            cas_numbers.extend(extract_cas_from_synonyms(compound_cid.synonyms))
        except Exception as e:
            print(f"Error retrieving CAS from CID {cid}: {e}")

    return cas_numbers if cas_numbers else None

def extract_cas_from_synonyms(synonyms):
    """Extracts CAS numbers from a list of synonyms.

    Args:
        synonyms (list): A list of synonyms for a compound.

    Returns:
        list: A list of CAS numbers extracted from the synonyms.
    """
    cas_numbers = []
    for synonym in synonyms:
        if synonym and len(synonym) <= 12 and synonym.count('-') == 2:  # Basic CAS number format check
            cas_numbers.append(synonym)
    return cas_numbers

# Function to process Excel file, convert CID, and save results
def process_excel(file_path, output_file, checkpoint_interval=100):
    """Processes an Excel file containing CIDs and retrieves CAS numbers for each CID.

    Args:
        file_path (str): Path to the input Excel file containing CIDs.
        output_file (str): Path to the output Excel file to save progress.
        checkpoint_interval (int): The interval (in rows) at which progress is saved.
    """
    
    # Check if output file exists (i.e., if progress was previously saved)
    if os.path.exists(output_file):
        # Load from the previously saved progress file
        df = pd.read_excel(output_file)
        print(f"Resuming from {output_file}")
    else:
        # Read the original input file
        df = pd.read_excel(file_path)
        # Initialize 'CAS_From_CID' and 'Processed' columns
        df['CAS_From_CID'] = None
        df['Processed'] = False
    
    # Loop through each row, skipping already processed rows
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        if row['Processed']:
            continue  # Skip rows that are already processed
        
        cid = row.get('CID')
        try:
            cas_numbers = get_cas_from_cid(cid=cid)

            if cas_numbers:
                df.at[index, 'CAS_From_CID'] = ', '.join(cas_numbers)

            # Mark the row as processed
            df.at[index, 'Processed'] = True

        except Exception as e:
            print(f"Error processing index {index} (CID: {cid}): {e}")

        # Save progress at regular intervals
        if index % checkpoint_interval == 0:
            df.to_excel(output_file, index=False)
            print(f"Checkpoint saved at row {index}.")

    # Final save after all rows are processed
    df.to_excel(output_file, index=False)
    print(f"CAS numbers have been written to {output_file}")

# Example usage
file_path = "filtered_compounds.xlsx"  # Replace with your input Excel file path
output_file = "output_with_cas.xlsx"  # Output file where CAS numbers will be written

# Process the Excel file and add CAS numbers with checkpoint saving
process_excel(file_path, output_file, checkpoint_interval=100)