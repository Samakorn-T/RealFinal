import time
import pandas as pd
import requests
from tqdm import tqdm  # For progress tracking
from requests.exceptions import RequestException, HTTPError

# Function to perform HTTP request with retry mechanism
def fetch_with_retries(url, retries=5, backoff_factor=1):
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes (400, 500, etc.)
            return response  # Return the successful response
        except HTTPError as e:
            if response.status_code == 400:  # Specific handling for Bad Request (invalid input)
                print(f"Bad request for URL: {url}. Skipping...")
                return None  # Skip further attempts if input is invalid
            else:
                print(f"HTTP error on attempt {attempt + 1}: {e}")
        except (RequestException, ConnectionError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
            else:
                raise  # Reraise the error if all retries fail
    return None

# Function to convert CID to CAS number
def convert_cid_to_cas(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/RN/JSON"
    response = fetch_with_retries(url)
    
    if response and response.status_code == 200:
        data = response.json()
        if 'InformationList' in data and 'Information' in data['InformationList']:
            cas_numbers = data['InformationList']['Information'][0].get('RN', [])
            return cas_numbers if cas_numbers else None
    return None

# Function to process Excel file, convert CID to CAS, and save results
def process_excel(file_path, output_file):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Add new column for CAS numbers from CID
    df['CAS_From_CID'] = None
    
    # Loop through each row in the DataFrame with progress tracking
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # Get CAS number from CID
        cid = row.get('CID')
        if pd.notna(cid):  # Check if CID is not NaN
            try:
                cas_numbers = convert_cid_to_cas(int(cid))  # Ensure CID is an integer
                df.at[index, 'CAS_From_CID'] = ', '.join(cas_numbers) if cas_numbers else None
            except Exception as e:
                print(f"Error processing CID at index {index}: {e}")

    # Write the updated DataFrame back to a new Excel file
    df.to_excel(output_file, index=False)
    print(f"CAS numbers have been written to {output_file}")

# Example usage
file_path = "filtered_compounds.xlsx"  # Replace with your input Excel file path
output_file = "output_with_cas.xlsx"  # Output file where CAS numbers will be written

# Process the Excel file and add CAS numbers
process_excel(file_path, output_file)