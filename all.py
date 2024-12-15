# External Imports
import os
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
# Internal Imports
from load_data import load_data, langs, label_map

# Load data from our files
datasets = load_data()


number_of_tags = len(label_map)
id2label = {v: k for k, v in label_map.items()}

# Combine all training datasets into one for vocabulary building
train_sets = [datasets[l]['train'] for l in langs]
combined_train = concatenate_datasets(train_sets)

# Build vocabulary from training tokens
word_counter = Counter()
for example in combined_train:
    word_counter.update(example['tokens'])
words = list(word_counter.keys())
words = ["<PAD>", "<UNK>"] + words
word2id = {w: i for i, w in enumerate(words)}
vocab_size = len(word2id)
print(f"Vocab size: {vocab_size}")

# Create combined datasets
def make_multilingual_dataset(datasets, langs):
    train_dataset = concatenate_datasets([datasets[l]['train'] for l in langs])
    val_dataset = concatenate_datasets([datasets[l]['validation'] for l in langs])
    test_dataset = concatenate_datasets([datasets[l]['test'] for l in langs])
    return DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})

multilingual_dataset = make_multilingual_dataset(datasets, langs)
def encode_tokens(tokens, max_len=50):
    # Convert tokens to IDs, unknown if not in vocab
    token_ids = [word2id[w] if w in word2id else word2id["<UNK>"] for w in tokens]
    token_ids = token_ids[:max_len]
    # Pad if shorter
    pad_len = max_len - len(token_ids)
    token_ids += [word2id["<PAD>"]] * pad_len
    return token_ids

def encode_tags(tags, max_len=50):
    tag_ids = tags[:max_len]
    pad_len = max_len - len(tag_ids)
    tag_ids += [label_map["O"]] * pad_len
    return tag_ids

class NERDataset(Dataset):
    def __init__(self, hf_dataset, max_len=50):
        self.hf_dataset = hf_dataset
        self.max_len = max_len

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        # Encode
        token_ids = encode_tokens(tokens, self.max_len)
        tag_ids = encode_tags(ner_tags, self.max_len)

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long)

train_data = NERDataset(multilingual_dataset['train'], max_len=50)
val_data = NERDataset(multilingual_dataset['validation'], max_len=50)
test_data = NERDataset(multilingual_dataset['test'], max_len=50)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

#######################################################################
#  MODELS                                                             #
#######################################################################
#########################################
# CRF Layer Implementation
#########################################
class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # No constraints for simplicity, but you can add if needed.

    def forward_alg(self, feats):
        batch_size, seq_len, num_tags = feats.size()
        alpha = feats[:, 0]  # shape: (batch_size, num_tags)
        for t in range(1, seq_len):
            # shape: (batch_size, num_tags, num_tags)
            score_t = alpha.unsqueeze(2) + self.transitions + feats[:, t].unsqueeze(1)
            alpha = torch.logsumexp(score_t, dim=1)
        return torch.logsumexp(alpha, dim=1)

    def score_sentence(self, feats, tags):
        batch_size, seq_len, num_tags = feats.size()
        # score start
        score = feats[torch.arange(batch_size), 0, tags[:, 0]]
        # accumulate transitions
        for t in range(1, seq_len):
            score += self.transitions[tags[:, t - 1], tags[:, t]] + feats[torch.arange(batch_size), t, tags[:, t]]
        return score

    def viterbi_decode(self, feats):
        batch_size, seq_len, num_tags = feats.size()
        backpointers = []
        viterbi_vars = feats[:, 0]
        for t in range(1, seq_len):
            next_vars = viterbi_vars.unsqueeze(2) + self.transitions + feats[:, t].unsqueeze(1)
            best_vars, best_idx = torch.max(next_vars, dim=1)
            viterbi_vars = best_vars
            backpointers.append(best_idx)

        best_path = []
        best_tag = torch.argmax(viterbi_vars, dim=1)
        best_path.append(best_tag.unsqueeze(1))
        for backptrs in reversed(backpointers):
            best_tag = torch.gather(backptrs, 1, best_tag.unsqueeze(1)).squeeze(1)
            best_path.append(best_tag.unsqueeze(1))
        best_path.reverse()
        best_path = torch.cat(best_path, dim=1)
        return best_path

    def neg_log_likelihood(self, feats, tags):
        forward_score = self.forward_alg(feats)
        gold_score = self.score_sentence(feats, tags)
        return (forward_score - gold_score).mean()


#########################################
# LSTM-CRF Model
#########################################
class LSTMCRF(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super(LSTMCRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2id["<PAD>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim*2, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, token_ids, tags=None):
        embeds = self.embedding(token_ids)
        lstm_out, _ = self.lstm(embeds)
        feats = self.linear(lstm_out)

        if tags is not None:
            loss = self.crf.neg_log_likelihood(feats, tags)
            return loss
        else:
            # decode
            return self.crf.viterbi_decode(feats)

lstmcrf_model = LSTMCRF(vocab_size=vocab_size, embed_dim=100, hidden_dim=128, num_tags=number_of_tags)
optimizer = torch.optim.Adam(lstmcrf_model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstmcrf_model.to(device)

######## Training
epochs = 1
for epoch in range(epochs):
    lstmcrf_model.train()
    total_loss = 0
    for token_ids, tag_ids in train_loader:
        token_ids, tag_ids = token_ids.to(device), tag_ids.to(device)
        optimizer.zero_grad()
        loss = lstmcrf_model(token_ids, tags=tag_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
lstmcrf_model.eval()

#########################################
# HMM Model
#########################################
class HMM:
    def __init__(self, label_map, vocab_size):
        self.label_map = label_map
        self.num_tags = len(label_map)
        self.vocab_size = vocab_size
        self.start_probs = np.zeros(self.num_tags)  # P(tag_start)
        self.transition_probs = np.zeros((self.num_tags, self.num_tags))  # P(tag_next | tag_current)
        self.emission_probs = np.zeros((self.num_tags, vocab_size))  # P(word | tag)

    def train(self, train_loader):
        # Counters for probabilities
        start_counter = Counter()
        transition_counter = Counter()
        emission_counter = Counter()

        # Iterate through training data
        for token_ids, tag_ids in train_loader:
            token_ids = token_ids.numpy()
            tag_ids = tag_ids.numpy()
            for tokens, tags in zip(token_ids, tag_ids):
                # Start probabilities
                start_counter[tags[0]] += 1

                # Transition and emission probabilities
                for i in range(len(tokens) - 1):
                    current_tag = tags[i]
                    next_tag = tags[i + 1]
                    token = tokens[i]
                    transition_counter[(current_tag, next_tag)] += 1
                    emission_counter[(current_tag, token)] += 1

                # Last word's emission
                emission_counter[(tags[-1], tokens[-1])] += 1

        # Convert counts to probabilities
        total_starts = sum(start_counter.values())
        self.start_probs = np.array([start_counter[tag] / total_starts for tag in range(self.num_tags)])

        for (tag, next_tag), count in transition_counter.items():
            self.transition_probs[tag, next_tag] = count
        self.transition_probs = self.transition_probs / self.transition_probs.sum(axis=1, keepdims=True)

        for (tag, token), count in emission_counter.items():
            self.emission_probs[tag, token] = count
        self.emission_probs = self.emission_probs / self.emission_probs.sum(axis=1, keepdims=True)

    def viterbi_decode(self, token_ids):
        seq_len = len(token_ids)
        viterbi = np.zeros((seq_len, self.num_tags))
        backpointer = np.zeros((seq_len, self.num_tags), dtype=int)

        # Initialize
        viterbi[0] = self.start_probs * self.emission_probs[:, token_ids[0]]

        # Recursion
        for t in range(1, seq_len):
            for tag in range(self.num_tags):
                prob_transitions = viterbi[t - 1] * self.transition_probs[:, tag]
                viterbi[t, tag] = np.max(prob_transitions) * self.emission_probs[tag, token_ids[t]]
                backpointer[t, tag] = np.argmax(prob_transitions)

        # Backtracking
        best_path = []
        best_tag = np.argmax(viterbi[-1])
        best_path.append(best_tag)

        for t in range(seq_len - 1, 0, -1):
            best_tag = backpointer[t, best_tag]
            best_path.insert(0, best_tag)

        return best_path

    def predict(self, data_loader):
        all_preds = []
        all_labels = []
        for token_ids, tag_ids in data_loader:
            token_ids = token_ids.numpy()
            tag_ids = tag_ids.numpy()
            for tokens, tags in zip(token_ids, tag_ids):
                preds = self.viterbi_decode(tokens)
                all_preds.append(preds)
                all_labels.append(tags.tolist())
        return all_preds, all_labels

######## Training
hmm_model = HMM(label_map=label_map, vocab_size=vocab_size)
hmm_model.train(train_loader)
hmm_preds, hmm_labels = hmm_model.predict(val_loader)

#######################################################################
#  MODEL EVALUATIONS                                                  #
#######################################################################
def evaluate_hmm(preds, labels, ignore_label=label_map["O"]):
    t = [x for seq in labels for x in seq]
    p = [x for seq in preds for x in seq]

    # Remove padding
    paired = [(x, y) for x, y in zip(t, p) if x != 0 or y != 0]

    t = [x for x, y in paired]
    p = [y for x, y in paired]

    tp = sum(1 for x, y in zip(t, p) if x == y and x != ignore_label)
    fp = sum(1 for x, y in zip(t, p) if y != x and y != ignore_label)
    fn = sum(1 for x, y in zip(t, p) if x != y and x != ignore_label)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return precision, recall, f1

def evaluate_lstmcrf(data_loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for token_ids, tag_ids in data_loader:
            token_ids, tag_ids = token_ids.to(device), tag_ids.to(device)
            preds = lstmcrf_model(token_ids)
            # preds shape: (batch, seq_len)
            # tag_ids shape: (batch, seq_len)
            preds = preds.cpu().tolist()
            true = tag_ids.cpu().tolist()

            for p_seq, t_seq in zip(preds, true):
                # Remove padding
                # pad is <PAD> token, but tags were padded with label_map["O"]
                # find first padding in token_ids if needed or just strip trailing PAD words:
                if label_map["O"] == 0:
                    # We know O is 0, so padded tags are also 0
                    # We'll ignore trailing PADs/O
                    while len(p_seq) > 0 and p_seq[-1] == 0 and t_seq[-1] == 0:
                        p_seq.pop()
                        t_seq.pop()
                all_preds.append(p_seq)
                all_labels.append(t_seq)

    # Compute precision, recall, f1
    t = [x for seq in all_labels for x in seq]
    p = [x for seq in all_preds for x in seq]

    # Ignore O for metrics
    ignore_label = label_map["O"]
    paired = [(x,y) for x,y in zip(t,p) if x!=ignore_label or y!=ignore_label]

    t = [x for x,y in paired]
    p = [y for x,y in paired]

    tp = sum(1 for x,y in zip(t,p) if x==y and x!=ignore_label)
    fp = sum(1 for x,y in zip(t,p) if y!=x and y!=ignore_label)
    fn = sum(1 for x,y in zip(t,p) if x!=y and x!=ignore_label)

    precision = tp/(tp+fp+1e-10)
    recall = tp/(tp+fn+1e-10)
    f1 = 2*(precision*recall)/(precision+recall+1e-10)
    return precision, recall, f1

hmm_precision, hmm_recall, hmm_f1 = evaluate_hmm(hmm_preds, hmm_labels)
print(f"[  HMM   ] Precision:{hmm_precision:.3f}, Recall:{hmm_recall:.3f}, F1:{hmm_f1:.3f}")

lstmcrf_precision, lstmcrf_recall, lstmcrf_f1 = evaluate_lstmcrf(val_loader)
print(f"[LSTM-CRF] Precision={lstmcrf_precision:.3f}, Recall={lstmcrf_recall:.3f}, F1={lstmcrf_f1:.3f}")

