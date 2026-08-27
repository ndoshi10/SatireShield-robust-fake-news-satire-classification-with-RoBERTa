from transformers import AutoModelForSequenceClassification

num_labels = len(le.classes_)  # from your label encoder

model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy
