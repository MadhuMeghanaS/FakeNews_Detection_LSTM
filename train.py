"""
Fake News Detection with a Bidirectional LSTM (PyTorch)
Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

"""

import argparse
import json
import os
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"


# Data utilities
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(data_dir: str) -> pd.DataFrame:
    fake = pd.read_csv(os.path.join(data_dir, "Fake.csv"))
    real = pd.read_csv(os.path.join(data_dir, "True.csv"))
    fake["label"] = 0
    real["label"] = 1
    df = pd.concat([fake, real], ignore_index=True)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)
    df = df[df["content"].str.len() > 0].reset_index(drop=True)
    return df[["content", "label"]]


def build_vocab(texts, min_freq: int = 5, max_size: int = 30000):
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.most_common(max_size):
        if freq < min_freq:
            continue
        vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int):
    tokens = text.split()[:max_len]
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens]
    ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


# Model
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        emb = self.embedding(x)                  # (B, T, E)
        out, (h_n, _) = self.lstm(emb)            # h_n: (2*layers, B, H)
        # concat final forward + backward hidden states from the last layer
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)       # (B, 2H)
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        logits = self.fc2(h).squeeze(1)
        return logits


# Train / eval loops
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy().astype(int))

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    df = load_data(args.data_dir)
    print(f"Total samples: {len(df)}  |  class balance:\n{df['label'].value_counts()}")

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)

    print("Building vocabulary...")
    vocab = build_vocab(train_df["content"].tolist(), min_freq=5)
    print(f"Vocab size: {len(vocab)}")

    train_ds = NewsDataset(train_df["content"].tolist(), train_df["label"].tolist(), vocab, args.max_len)
    val_ds = NewsDataset(val_df["content"].tolist(), val_df["label"].tolist(), vocab, args.max_len)
    test_ds = NewsDataset(test_df["content"].tolist(), test_df["label"].tolist(), vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = BiLSTMClassifier(len(vocab), args.embed_dim, args.hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model.pt")
            with open("vocab.json", "w") as f:
                json.dump(vocab, f)

    # Final test evaluation using the best checkpoint
    model.load_state_dict(torch.load("model.pt"))
    test_loss, test_acc, preds, labels = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print("\n Test set results ")
    print(f"Accuracy: {test_acc:.4f} | F1: {f1_score(labels, preds):.4f}")
    print(classification_report(labels, preds, target_names=["Fake", "Real"]))
    print("Confusion matrix:\n", confusion_matrix(labels, preds))
    print("\nSaved best model to model.pt and vocabulary to vocab.json")


if __name__ == "__main__":
    main()
