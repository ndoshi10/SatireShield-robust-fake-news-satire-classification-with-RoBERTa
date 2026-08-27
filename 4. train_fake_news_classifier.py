import pandas as pd
import json
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# ================== 1. Load All Datasets ==================

# Load ISOT dataset
df_fake = pd.read_csv('/content/Fake.csv')
df_fake['label'] = 'fake'
df_true = pd.read_csv('/content/True.csv')
df_true['label'] = 'real'
isot_df = pd.concat([df_fake, df_true], ignore_index=True)
# It seems the first notebook cell intended to use 'text' and 'title' separately,
# but the second and third cells assumed 'text' existed or combined them later.
# To be consistent with the second notebook cell that failed, let's assume 'text'
# is the column, or combine 'title' and 'text' if 'text' doesn't exist alone.
# Checking the original data columns: Fake.csv and True.csv have 'title' and 'text'.
# Let's combine them for a richer text representation.
isot_df['text'] = isot_df['title'].fillna('') + ' ' + isot_df['text'].fillna('')


# Load LIAR dataset
columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state',
           'party', 'barely_true_counts', 'false_counts', 'pants_on_fire_counts', 'context']
train_path = '/content/train.tsv'
valid_path = '/content/valid.tsv'
test_path = '/content/test.tsv'
liar_df = pd.concat([
    pd.read_csv(train_path, sep='\t', names=columns, header=None),
    pd.read_csv(valid_path, sep='\t', names=columns, header=None),
    pd.read_csv(test_path, sep='\t', names=columns, header=None)
], ignore_index=True)
# Map detailed LIAR labels to a unified set
liar_label_map = {
    'pants-fire': 'fake',
    'false': 'fake',
    'barely-true': 'fake',
    'half-true': 'biased',
    'mostly-true': 'biased',
    'true': 'real'
}
liar_df['label'] = liar_df['label'].map(liar_label_map)
liar_df = liar_df[['statement', 'label']].rename(columns={'statement': 'text'})
# Drop rows where label mapping resulted in NaN (if any unmapped labels existed)
liar_df.dropna(subset=['label'], inplace=True)


# Load Satire dataset
satire_path = '/content/News_Category_Dataset_v3.json'
satire_data = []
with open(satire_path, 'r') as f:
    for line in f:
        if line.strip():
            try:
                satire_data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
satire_df_full = pd.json_normalize(satire_data)

# Correct filtering for satire categories, using the list defined earlier
satire_categories = ['SATIRE', 'satire', 'Satire']
satire_df = satire_df_full[satire_df_full['category'].isin(satire_categories)].copy()

# Combine headline and short_description for satire text
satire_df['text'] = satire_df['headline'].fillna('') + ' ' + satire_df['short_description'].fillna('')
satire_df['label'] = 'satire' # Assign 'satire' label to these filtered rows

# Keep other non-satire categories from the original satire_df_full if needed,
# or just combine the satire subset with other datasets.
# For this fix, we assume we only want 'satire' labeled data from this source.
# If you want to keep other categories as 'real' from this dataset, you would add them here.


# ================== 2. Combine and Clean ==================

# Select only 'text' and 'label' columns from each source before concatenating
combined_df = pd.concat([
    isot_df[['text', 'label']],
    liar_df[['text', 'label']],
    satire_df[['text', 'label']] # Ensure satire_df only has text and label
], ignore_index=True)

# Remove any rows that might have ended up with NaN in 'text' or 'label' after joins/filters
combined_df.dropna(subset=['text', 'label'], inplace=True)


# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Updated regex to also include digits since they were in the original code #4
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

combined_df['clean_text'] = combined_df['text'].astype(str).apply(clean_text)
# Drop rows where clean_text might have become empty after cleaning if they exist
combined_df = combined_df[combined_df['clean_text'] != ''].copy()


# ================== 3. Encode Labels ==================

label_encoder = LabelEncoder()
combined_df['label_encoded'] = label_encoder.fit_transform(combined_df['label'])
num_labels = len(label_encoder.classes_)

print("Label mapping:")
for i, class_ in enumerate(label_encoder.classes_):
    print(f"{class_} -> {i}")

print("\nClass distribution BEFORE handling rare classes:")
print(combined_df['label'].value_counts())


# --- Crucial Step: Identify and Handle Rare Classes ---
# Find classes with less than 2 samples
class_counts = combined_df['label_encoded'].value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()

if rare_classes:
    print(f"\nFound rare classes with < 2 samples: {rare_classes}")
    # Get the original label names for rare classes
    rare_labels = label_encoder.inverse_transform(rare_classes)
    print(f"Corresponding labels: {list(rare_labels)}")

    # Option 1: Remove samples belonging to rare classes
    # This is the simplest solution for train_test_split with stratify
    combined_df_filtered = combined_df[~combined_df['label_encoded'].isin(rare_classes)].copy()
    print(f"\nRemoved {len(combined_df) - len(combined_df_filtered)} samples belonging to rare classes.")
    combined_df = combined_df_filtered

    # Re-encode labels if you removed classes, otherwise skip this re-encoding
    # if the removed classes were few and didn't change the set of remaining classes significantly.
    # However, it's safer to re-encode if you remove classes to get contiguous label indices.
    label_encoder = LabelEncoder()
    combined_df['label_encoded'] = label_encoder.fit_transform(combined_df['label'])
    num_labels = len(label_encoder.classes_)
    print("\nLabel mapping AFTER removing rare classes:")
    for i, class_ in enumerate(label_encoder.classes_):
        print(f"{class_} -> {i}")
    print("\nClass distribution AFTER handling rare classes:")
    print(combined_df['label'].value_counts())

    # If after removing rare classes, the number of labels changed, you need to update num_labels
    num_labels = len(label_encoder.classes_)

else:
    print("\nNo rare classes with < 2 samples found.")

# Ensure there are at least 2 samples per class remaining before splitting
if combined_df['label_encoded'].value_counts().min() < 2:
     raise ValueError("Still have classes with less than 2 members after filtering. Check data.")


# ================== 4. Train-Test Split ==================

# Now perform the split on the potentially filtered DataFrame
train_df, test_df = train_test_split(
    combined_df,
    test_size=0.2,
    stratify=combined_df['label_encoded'], # Stratify using the updated encoded labels
    random_state=42
)

print(f"\nTrain size: {train_df.shape[0]} samples")
print(f"Test size: {test_df.shape[0]} samples")

# ================== 5. Tokenization and Dataloaders ==================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Changed tokenizer and model to roberta-base to match notebook cell #5
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=num_labels # Use the potentially updated num_labels
).to(device)

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_batch(batch):
    texts, labels = zip(*batch)
    # Use the tokenizer defined above (roberta-base)
    encodings = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    encodings['labels'] = torch.tensor(labels)
    return {k: v.to(device) for k, v in encodings.items()}

# Use the full train_df and test_df from the split (or keep slicing if memory is an issue)
# Keep the slicing from your original code if you intended to work with a smaller subset
train_texts = list(train_df['clean_text']) #[:2000] # Uncomment slicing if needed
train_labels = list(train_df['label_encoded']) #[:2000] # Uncomment slicing if needed
test_texts = list(test_df['clean_text']) #[:500] # Uncomment slicing if needed
test_labels = list(test_df['label_encoded']) #[:500] # Uncomment slicing if needed

train_dataset = NewsDataset(train_texts, train_labels)
test_dataset = NewsDataset(test_texts, test_labels)

# Use the batch size from your original code block #4 (16)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)

# ================== 6. Train the Model ==================

optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    # device is already defined globally
    for batch in tqdm(dataloader):
        # batch items are already on device from collate_batch
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0 # Added total_loss tracking for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader):
             # batch items are already on device from collate_batch
            outputs = model(**batch)
            loss = outputs.loss # Added loss calculation for evaluation
            total_loss += loss.item() # Accumulate evaluation loss
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    # Return both loss and accuracy from evaluation
    return total_loss / len(dataloader), correct / total

epochs = 2 # Set number of epochs
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, train_loader, optimizer)
    # Updated eval_epoch call to receive both loss and accuracy
    eval_loss, accuracy = eval_epoch(model, test_loader)
    # Updated print statement to show eval loss as well
    print(f"Train Loss: {train_loss:.4f} | Test Loss: {eval_loss:.4f} | Test Accuracy: {accuracy:.4f}")
    scheduler.step()