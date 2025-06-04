import pandas as pd

# File paths
input_csv = 'diabetic_data.csv'
output_with_weight = 'diabetic_with_weight.csv'
output_without_weight = 'diabetic_without_weight.csv'

# Columns to drop from all outputs
columns_to_drop = [
    'encounter_id',
    'patient_nbr',
    'payer_code',
    'medical_specialty',
    'citoglipton'
]

# Columns that must NOT be missing
critical_columns = ['race', 'diag_1', 'diag_2', 'diag_3']

# Read the CSV, treating '?' as missing values
# Remove na_filter=False to allow proper NaN conversion
df = pd.read_csv(input_csv, na_values=['?'], keep_default_na=False, low_memory=False)

# Drop unwanted columns
df = df.drop(columns=columns_to_drop, errors='ignore')

# Drop rows with missing values in critical columns
df_clean = df.dropna(subset=critical_columns)

# Split into two datasets based on presence of weight
# Check for both NaN and '?' string values to be safe
df_with_weight = df_clean[
    (df_clean['weight'].notna()) & 
    (df_clean['weight'] != '?') & 
    (df_clean['weight'] != '')
].copy()

df_without_weight = df_clean[
    (df_clean['weight'].isna()) | 
    (df_clean['weight'] == '?') | 
    (df_clean['weight'] == '')
].copy()

# Drop the 'weight' column from the "without weight" DataFrame
df_without_weight = df_without_weight.drop(columns=['weight'], errors='ignore')

# Save to CSV
df_with_weight.to_csv(output_with_weight, index=False, na_rep='')
df_without_weight.to_csv(output_without_weight, index=False, na_rep='')

print(f"Rows with weight: {len(df_with_weight)}")
print(f"Rows without weight: {len(df_without_weight)}")
