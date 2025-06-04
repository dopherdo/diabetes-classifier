import pandas as pd

# File paths
input_csv = 'diabetic_data.csv'
output_with_weight = 'diabetic_with_weight.csv'
output_without_weight = 'diabetic_without_weight.csv'

# Columns to drop entirely
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
df = pd.read_csv(input_csv, na_values=['?'])

# Drop unwanted columns
df = df.drop(columns=columns_to_drop, errors='ignore')

# Drop rows with missing values in critical columns
df_clean = df.dropna(subset=critical_columns)

# Split into two datasets based on presence of weight
df_with_weight = df_clean[df_clean['weight'].notna()]
df_without_weight = df_clean[df_clean['weight'].isna()]

# Save to CSV
df_with_weight.to_csv(output_with_weight, index=False)
df_without_weight.to_csv(output_without_weight, index=False)

print(f"Saved {len(df_with_weight)} rows with weight to: {output_with_weight}")
print(f"Saved {len(df_without_weight)} rows without weight to: {output_without_weight}")
