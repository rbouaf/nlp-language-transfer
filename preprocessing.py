print("Please wait...")
import threading
import itertools
import sys
import time
from datasets import concatenate_datasets, DatasetDict
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter,defaultdict
#internal imports
from load_data import load_data
from langs import langs, label_map
from load_data import load_all_data
import numpy as np
from transformers import DistilBertTokenizerFast

number_of_tags = len(label_map)
id2label = {v: k for k, v in label_map.items()}
# Load data from our files
datasets = load_all_data()
def vocab_and_datasets(langs):
    # Combine all training datasets for vocabulary building
    train_sets = [datasets[l]['train'] for l in langs]
    combined_train = concatenate_datasets(train_sets)
    word_counter = Counter()
    for example in combined_train:# Build vocabulary from training tokens
        word_counter.update(example['tokens'])

    words = list(word_counter.keys())
    words = ["<PAD>", "<UNK>"] + words
    word2id = {w: i for i, w in enumerate(words)}
    vocab_size = len(word2id)
    print(f"Vocab size: {vocab_size}")

    # Create combined datasets
    train_dataset = concatenate_datasets([datasets[l]['train'] for l in langs])
    val_dataset = concatenate_datasets([datasets[l]['validation'] for l in langs])
    test_dataset = concatenate_datasets([datasets[l]['test'] for l in langs])
    multilingual_dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})
    return word2id,vocab_size,multilingual_dataset

# word2id,vocab_size,multilingual_dataset = vocab_and_datasets(langs)

def preprocess(multilingual_dataset, word2id):
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
    return train_data,val_data,test_data,train_loader,val_loader,test_loader

def build_feature_mapping(datasets):
    """Build a mapping for categorical features to numerical IDs."""
    feature_mapping = {}
    for split in ["train", "validation", "test"]:
        for example in datasets[split]:
            tokens = example["tokens"]
            for idx in range(len(tokens)):
                features = extract_features(tokens, idx)
                for key, value in features.items():
                    if isinstance(value, str):  # Handle categorical features
                        if key not in feature_mapping:
                            feature_mapping[key] = {}
                        if value not in feature_mapping[key]:
                            feature_mapping[key][value] = len(feature_mapping[key]) + 1
    return feature_mapping

def extract_features(tokens, idx):
    """Extract handcrafted features for a single token at position idx."""
    features = {}
    # Current token features
    features["token"] = tokens[idx].lower()
    features["token_len"] = len(tokens[idx])
    features["is_capitalized"] = tokens[idx][0].isupper()
    features["is_digit"] = tokens[idx].isdigit()

    # Prefix and suffix features
    features["prefix_1"] = tokens[idx][:1]
    features["prefix_2"] = tokens[idx][:2]
    features["suffix_1"] = tokens[idx][-1:]
    features["suffix_2"] = tokens[idx][-2:]

    # Surrounding tokens (context window)
    features["prev_token"] = tokens[idx - 1].lower() if idx > 0 else "<PAD>"
    features["next_token"] = tokens[idx + 1].lower() if idx < len(tokens) - 1 else "<PAD>"

    return features

def prepare_data_for_decision_tree(hf_dataset, feature_mapping):
    """Prepare data specifically for Decision Tree models."""
    X, y = [], []
    for example in hf_dataset:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        for idx in range(len(tokens)):
            features = extract_features(tokens, idx)
            vector = [
                feature_mapping[key].get(features.get(key, ""), 0)
                if isinstance(features.get(key, ""), str)
                else features.get(key, 0)
                for key in feature_mapping
            ]
            X.append(vector)
            y.append(ner_tags[idx])
    return np.array(X), np.array(y)


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')
def tokenize_and_align_labels(examples):
    """Tokenize inputs and align NER labels with word-piece tokens."""
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        padding='max_length',
        is_split_into_words=True,
        max_length=50  # Adjust if needed
    )
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their word IDs
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Special token
            elif word_id != previous_word_id:
                label_ids.append(label[word_id])  # Label for the first token of the word
            else:
                label_ids.append(-100)  # Label the rest of the tokens of the word as -100
            previous_word_id = word_id
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def preprocess_for_distilbert(multilingual_dataset):
    """Preprocess datasets for DistilBERT."""
    return multilingual_dataset.map(tokenize_and_align_labels, batched=True)

def compute_unigram_counts(dataset):
    """Compute unigram counts for a dataset."""
    unigram_counts = defaultdict(int)
    for example in dataset['train']:
        for token in example['tokens']:
            unigram_counts[token] += 1
    return unigram_counts