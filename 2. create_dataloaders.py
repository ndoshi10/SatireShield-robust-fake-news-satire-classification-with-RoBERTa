import torch
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
max_length = 128

def collate_batch(batch):
    texts, labels = zip(*batch)
    encodings = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    labels = torch.tensor(labels)
    encodings['labels'] = labels
    return encodings

from torch.utils.data import DataLoader

train_dataset = NewsDataset(list(train_df['clean_text']), list(train_df['label_encoded']))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)

test_dataset = NewsDataset(list(test_df['clean_text']), list(test_df['label_encoded']))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)
