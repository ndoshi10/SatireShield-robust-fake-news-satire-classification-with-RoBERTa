#1
import pandas as pd
import json

# -----------------------------------
# Load ISOT Fake and True News CSV files
# -----------------------------------

isot_file1 = '/content/Fake.csv'
isot_file2 = '/content/True.csv'

df_fake = pd.read_csv(isot_file1)
df_true = pd.read_csv(isot_file2)

# Add a label column to each (if not already present)
df_fake['label'] = 'fake'
df_true['label'] = 'real'

# Combine datasets
isot_df = pd.concat([df_fake, df_true], ignore_index=True)

print(f"=== ISOT Dataset Loaded ===")
print(f"Total records: {isot_df.shape[0]}")
display(isot_df.head())
print(f"Columns: {list(isot_df.columns)}\n")

# -----------------------------------
# Load LIAR dataset (train, valid, test)
# -----------------------------------

liar_columns = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state',
    'party', 'barely_true_counts', 'false_counts', 'pants_on_fire_counts', 'context'
]

train_path = '/content/train.tsv'
valid_path = '/content/valid.tsv'
test_path = '/content/test.tsv'

train_df = pd.read_csv(train_path, sep='\t', names=liar_columns, header=None)
valid_df = pd.read_csv(valid_path, sep='\t', names=liar_columns, header=None)
test_df = pd.read_csv(test_path, sep='\t', names=liar_columns, header=None)

# Combine LIAR splits
liar_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

print(f"=== LIAR Dataset Loaded ===")
print(f"Total records: {liar_df.shape[0]}")
display(liar_df.head())
print(f"Columns: {list(liar_df.columns)}\n")

# -----------------------------------
# Load Satire Dataset (JSON lines format)
# -----------------------------------

satire_path = '/content/News_Category_Dataset_v3.json'

satire_data = []
with open(satire_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                satire_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line}\nError: {e}")

satire_df = pd.json_normalize(satire_data)

print(f"=== Satire Dataset Loaded ===")
print(f"Total records: {satire_df.shape[0]}")
display(satire_df.head())
print(f"Columns: {list(satire_df.columns)}")

#2
# Normalize ISOT labels (lowercase for safety)
isot_df['label'] = isot_df['label'].str.lower()

# For ISOT, usually text column is 'text' or 'title' + 'text' — check columns
# Let's assume 'text' column exists, if not adjust accordingly
print("ISOT text columns:", [col for col in isot_df.columns if 'text' in col.lower()])

# For LIAR dataset: map detailed labels to unified labels
liar_label_map = {
    'pants-fire': 'fake',
    'false': 'fake',
    'barely-true': 'fake',
    'half-true': 'biased',
    'mostly-true': 'biased',
    'true': 'real'
}

liar_df['label'] = liar_df['label'].map(liar_label_map)

# For LIAR, main text is in 'statement' column
print("LIAR text column: 'statement'")

# For Satire dataset: the dataset may have a 'category' column or similar
# Find categories that relate to satire
print("Satire dataset unique categories:", satire_df['category'].unique())

# Define which categories count as satire (this depends on the dataset; example below)
satire_categories = ['SATIRE', 'satire', 'Satire']

# Create label column for satire_df
satire_df['label'] = satire_df['category'].apply(lambda x: 'satire' if x in satire_categories else 'real')

# Main text column (based on dataset structure)
print("Satire text columns:", [col for col in satire_df.columns if 'headline' in col.lower() or 'text' in col.lower()])

# Example if text column is 'headline' or 'short_description'
# We'll concatenate if needed or pick one main text column

#3
# Prepare ISOT dataset: select text and label columns
isot_df_clean = isot_df[['text', 'label']].copy()

# Prepare LIAR dataset
liar_df_clean = liar_df[['statement', 'label']].copy()
liar_df_clean.rename(columns={'statement': 'text'}, inplace=True)

# Prepare Satire dataset
# Assuming 'headline' is the main text
satire_df_clean = satire_df[['headline', 'label']].copy()
satire_df_clean.rename(columns={'headline': 'text'}, inplace=True)

# Combine all datasets
combined_df = pd.concat([isot_df_clean, liar_df_clean, satire_df_clean], ignore_index=True)

print(f"Combined dataset shape: {combined_df.shape}")
print(combined_df['label'].value_counts())
display(combined_df.sample(10))

#4
import re

def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove special characters and digits (keep alphabets and spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply cleaning to the 'text' column
combined_df['clean_text'] = combined_df['text'].astype(str).apply(clean_text)

# Quick check
combined_df[['text', 'clean_text', 'label']].sample(5)

#5
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
combined_df['label_encoded'] = le.fit_transform(combined_df['label'])

print("Label mapping:")
for i, class_ in enumerate(le.classes_):
    print(f"{class_} -> {i}")

# Quick look
combined_df[['label', 'label_encoded']].head()

#6
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    combined_df, test_size=0.2, stratify=combined_df['label_encoded'], random_state=42
)

print(f"Train size: {train_df.shape[0]} samples")
print(f"Test size: {test_df.shape[0]} samples")
