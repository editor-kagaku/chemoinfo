import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import MolStandardize
from chembl_structure_pipeline import standardizer

# functions for compound standardization
def mol_standardize(mol):
    # standardization by ChEMBL structure pipeline
    std_molblock = standardizer.standardize_molblock(Chem.MolToMolBlock(mol))
    parent_molblock, _ = standardizer.get_parent_molblock(std_molblock)
    mol_norm = Chem.MolFromMolBlock(parent_molblock)
    # keep largest fragment
    lfc = MolStandardize.fragment.LargestFragmentChooser()
    mol_l = lfc.choose(mol_norm)
    return mol_l

def mol_to_standardized_smiles_from_smiles(s):
    try:
        return Chem.MolToSmiles(mol_standardize(Chem.MolFromSmiles(s)))
    except:
        return None

df = pd.read_csv("../data/activity_data.csv", usecols=['SMILES', 'r_avg_IC50', 'f_avg_IC50'])
df_active = df.query('not(r_avg_IC50 > 10 or f_avg_IC50 > 10)')[['SMILES']]
df_active['class'] = 1.0
df_inactive = df.query('r_avg_IC50 > 10 or f_avg_IC50 > 10')[['SMILES']]
df_inactive['class'] = 0.0
df_all = pd.concat([df_active, df_inactive])

df_all['SMILES'] = [mol_to_standardized_smiles_from_smiles(s) for s in df_all['SMILES']]

df_train, df_test, _, _ = train_test_split(df_all, df_all['class'], test_size=0.2, random_state=19, stratify=df_all['class'])
df_train.to_csv("../data/Mpro_train.csv", index=False)
df_test.to_csv("../data/Mpro_test.csv", index=False)
