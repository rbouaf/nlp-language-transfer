import torch
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import DataLoader
from transformers import AdamW
from datasets import DatasetDict
class LSTMCRF(nn.Module):
    def __init__(self, num_labels, embedding_dim=768, hidden_dim=256):
        super(LSTMCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        lstm_out, _ = self.lstm(input_ids)
        emissions = self.hidden2tag(lstm_out)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.type(torch.uint8))
            return loss
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask.type(torch.uint8))
            return predictions

# Prepare data loaders
def collate_fn(batch):
    input_ids = torch.tensor([x['input_ids'] for x in batch])
    attention_mask = torch.tensor([x['attention_mask'] for x in batch])
    labels = torch.tensor([x['labels'] for x in batch])
    return input_ids, attention_mask, labels

train_loader = DataLoader(encoded_dataset['train'], batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(encoded_dataset['validation'], batch_size=16, collate_fn=collate_fn)

# Initialize model and optimizer
model = LSTMCRF(num_labels=num_labels)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(5):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        loss = model(input_ids, attention_mask, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            loss = model(input_ids, attention_mask, labels)
            val_loss += loss.item()
    print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")