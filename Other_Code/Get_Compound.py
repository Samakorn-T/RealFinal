import pubchempy as pcp
import pandas as pd

def get_compounds_with_criteria():
    compounds = []
    total_searches = 7 * max(2 * c + 3 for c in range(1, 8)) * max(2 * c + 1 for c in range(1, 8))
    completed_searches = 0
    for c in range(1, 8):
        for h in range(1, 2 * c + 3):
            for o in range(0, 2 * c + 1):
                search_formula = f'C{c}H{h}' + (f'O{o}' if o > 0 else '')  # สร้างสูตรเคมีสำหรับการค้นหา
                try:
                    results = pcp.get_compounds(search_formula, 'formula')
                    if results:
                        for compound in results:
                            name = compound.iupac_name
                            molecular_formula = compound.molecular_formula  # ป้องกันการเขียนทับตัวแปร
                            smiles = compound.canonical_smiles
                            cid = compound.cid
                            compounds.append((name, molecular_formula, smiles, cid))
                    else:
                        print(f"No results found for {search_formula}")
                except Exception as e:
                    print(f"Error fetching data for {search_formula}: {e}")
                completed_searches += 1
                print(f"Processing: {search_formula} ({completed_searches}/{total_searches} searches completed, {100 * completed_searches / total_searches:.2f}%)")
    return compounds

compounds = get_compounds_with_criteria()

compounds_data = {
    'Name': [],
    'Formula': [],
    'SMILES': [],
    'CID': []
}

for compound in compounds:
    compounds_data['Name'].append(compound[0])
    compounds_data['Formula'].append(compound[1])
    compounds_data['SMILES'].append(compound[2])
    compounds_data['CID'].append(compound[3])

df = pd.DataFrame(compounds_data)
df.to_excel('compounds.xlsx', index=False)

print("Data has been exported to compounds.xlsx")