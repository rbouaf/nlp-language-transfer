print("Importing...")
import torch.nn as nn
import numpy as np
import re
import torch
from collections import Counter,defaultdict
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from collections import Counter, defaultdict
import numpy as np
from evaluate import load as load_metric
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification

# internal import
from preprocessing import prepare_data_for_decision_tree,compute_unigram_counts
from langs import label_map

#########################################
# HMM Model
#########################################
def hmm(vocab_size,train_loader,val_loader):
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

    t = [x for seq in hmm_labels for x in seq]
    p = [x for seq in hmm_preds for x in seq]

    # Remove padding
    paired = [(x, y) for x, y in zip(t, p) if x != 0 or y != 0]

    t = [x for x, y in paired]
    p = [y for x, y in paired]

    tp = sum(1 for x, y in zip(t, p) if x == y and x != label_map["O"])
    fp = sum(1 for x, y in zip(t, p) if y != x and y != label_map["O"])
    fn = sum(1 for x, y in zip(t, p) if x != y and x != label_map["O"])

    hmm_precision = tp / (tp + fp + 1e-10)
    hmm_recall = tp / (tp + fn + 1e-10)
    hmm_f1 = 2 * (hmm_precision * hmm_recall) / (hmm_precision + hmm_recall + 1e-10)
    print(f"[     HMM     ] Precision: {hmm_precision:.3f} \t Recall: {hmm_recall:.3f} \t F1: {hmm_f1:.3f}")
    return hmm_precision, hmm_recall, hmm_f1



#########################################
# LSTM-CRF Model
#########################################
################
# CRF Layer
################
def lstmcrf(vocab_size, train_loader, val_loader,word2id,number_of_tags):
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


# LSTM-CRF implementation
#####################
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

    lstmcrf_model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for token_ids, tag_ids in val_loader:
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

    lstmcrf_precision = tp/(tp+fp+1e-10)
    lstmcrf_recall = tp/(tp+fn+1e-10)
    lstmcrf_f1 = 2*(lstmcrf_precision*lstmcrf_recall)/(lstmcrf_precision+lstmcrf_recall+1e-10)
    print(f"      LSTM-CRF| Precision: {lstmcrf_precision:.3f} \t Recall: {lstmcrf_recall:.3f} \t F1: {lstmcrf_f1:.3f}")
    return lstmcrf_precision, lstmcrf_recall, lstmcrf_f1

#####################################################
# Bootstrapping with Seed Entities + Hearst Patterns#
#####################################################
def bootstrapping(id2label,val_loader):
    class BootstrappingNER:
        def __init__(self, label_map, seed_entities):
            self.label_map = label_map
            self.seed_entities = seed_entities
            self.patterns = self._define_patterns()

        def _define_patterns(self):
            """
            Define Hearst-like patterns for entity extraction.
            """
            return {
                "B-PER": [r"\b(?:Dr|Mr|Ms|Mrs|Prof)\.?\s+[A-Z][a-z]+",  # Titles followed by a name
                          r"\b[A-Z][a-z]+\s+[A-Z][a-z]+"],  # First name + Last name
                "B-ORG": [r"\b[A-Z][a-z]+ (Corporation|Company|Inc|Ltd)",  # Corporate names
                          r"\b[A-Z][a-z]+ (University|College)"],  # Educational institutions
                "B-LOC": [r"\b(?:City of|Town of|Province of)\s+[A-Z][a-z]+",  # Locations with a prefix
                          r"\b[A-Z][a-z]+,?\s+[A-Z][a-z]+"]  # City, Country
            }

        def extract_entities(self, text):
            """
            Extract entities from text using patterns and seed entities.
            """
            predictions = []
            for tag, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        predictions.append((match.start(), match.end(), tag, match.group()))
            return predictions

        def predict(self, data_loader):
            """
            Apply patterns to the dataset and predict entities.
            """
            all_preds = []
            all_labels = []

            for token_ids, tag_ids in data_loader:
                # Decode tokens back to text
                tokens = [[id2label[tag_id] for tag_id in sequence] for sequence in tag_ids.numpy()]
                for token_seq in tokens:
                    text = " ".join(token_seq)
                    predictions = self.extract_entities(text)
                    all_preds.append(predictions)
                    all_labels.append(token_seq)

            return all_preds, all_labels

    # Seed Entities
    seed_entities = {
    "B-PER": ["John Doe", "Jane Smith"],
    "B-ORG": ["OpenAI", "Google", "Stanford University"],
    "B-LOC": ["San Francisco", "New York"]
}

    # Bootstrapping model
    bootstrapper = BootstrappingNER(label_map, seed_entities)

    # Predict using bootstrapping
    bootstrapping_preds, bootstrapping_labels = bootstrapper.predict(val_loader)
    # Flatten predictions and labels
    t = [x for seq in bootstrapping_labels for x in seq]
    p = [x for seq in bootstrapping_preds for x in seq]

    # Ignore padding and O labels
    paired = [(x, y) for x, y in zip(t, p) if x != label_map["O"] or y != label_map["O"]]

    t = [x for x, y in paired]
    p = [y for x, y in paired]

    tp = sum(1 for x, y in zip(t, p) if x == y and x != label_map["O"])
    fp = sum(1 for x, y in zip(t, p) if y != x and y != label_map["O"])
    fn = sum(1 for x, y in zip(t, p) if x != y and x != label_map["O"])

    bootstrap_precision = tp / (tp + fp + 1e-10)
    bootstrap_recall = tp / (tp + fn + 1e-10)
    bootstrap_f1 = 2 * (bootstrap_precision * bootstrap_recall) / (bootstrap_precision + bootstrap_recall + 1e-10)
    print(
        f" Bootstrapping| Precision: {bootstrap_precision:.3f} \t Recall: {bootstrap_recall:.3f} \t F1: {bootstrap_f1:.3f}")

    return bootstrap_precision, bootstrap_recall, bootstrap_f1


#########################################
# Brown Clustering Model
# #########################################
# class BrownClustering:
#     def __init__(self, bigram_counts, unigram_counts, num_clusters):
#         self.bigram_counts = bigram_counts
#         self.unigram_counts = unigram_counts
#         self.num_clusters = num_clusters
#         self.word_to_cluster = {}
#         self.cluster_to_words = defaultdict(set)
#
#     def initialize_clusters(self):
#         for word in self.unigram_counts:
#             cluster_id = len(self.word_to_cluster)
#             self.word_to_cluster[word] = cluster_id
#             self.cluster_to_words[cluster_id].add(word)
#
#     def merge_clusters(self, cluster_a, cluster_b):
#         new_cluster_id = max(self.cluster_to_words) + 1
#         self.cluster_to_words[new_cluster_id] = self.cluster_to_words[cluster_a] | self.cluster_to_words[cluster_b]
#         for word in self.cluster_to_words[new_cluster_id]:
#             self.word_to_cluster[word] = new_cluster_id
#         del self.cluster_to_words[cluster_a]
#         del self.cluster_to_words[cluster_b]
#
#     def calculate_mutual_information(self, cluster_a, cluster_b):
#         total_count = sum(self.unigram_counts.values())
#         bigram_sum = sum(
#             self.bigram_counts[(word1, word2)]
#             for word1 in self.cluster_to_words[cluster_a]
#             for word2 in self.cluster_to_words[cluster_b]
#             if (word1, word2) in self.bigram_counts
#         )
#         p_a = sum(self.unigram_counts[word] for word in self.cluster_to_words[cluster_a]) / total_count
#         p_b = sum(self.unigram_counts[word] for word in self.cluster_to_words[cluster_b]) / total_count
#         p_ab = bigram_sum / total_count
#         return p_ab * (p_ab / (p_a * p_b + 1e-10))
#
#     def run(self):
#         self.initialize_clusters()
#         while len(self.cluster_to_words) > self.num_clusters:
#             max_mi = float('-inf')
#             best_pair = None
#             for cluster_a in self.cluster_to_words:
#                 for cluster_b in self.cluster_to_words:
#                     if cluster_a != cluster_b:
#                         mi = self.calculate_mutual_information(cluster_a, cluster_b)
#                         if mi > max_mi:
#                             max_mi = mi
#                             best_pair = (cluster_a, cluster_b)
#
#             if best_pair:
#                 self.merge_clusters(*best_pair)
#
#         return self.word_to_cluster
#
# class BrownClusteringPredictor:
#     def __init__(self, word_to_cluster):
#         self.word_to_cluster = word_to_cluster
#         self.cluster_tag_map = defaultdict(Counter)
#
#     def train(self, train_data):
#         for token_ids, tag_ids in train_data:
#             for token, tag in zip(token_ids, tag_ids):
#                 cluster_id = self.word_to_cluster.get(token, 0)
#                 self.cluster_tag_map[cluster_id][tag] += 1
#
#     def predict(self, token_ids):
#         return [
#             self.cluster_tag_map.get(self.word_to_cluster.get(token, 0), Counter()).most_common(1)[0][0]
#             for token in token_ids
#         ]
# def compute_ngram_statistics(data_loader):
#     bigram_counts = defaultdict(int)
#     unigram_counts = defaultdict(int)
#     for token_batch, _ in data_loader:  # Unpack token IDs and ignore tag IDs
#         token_batch = token_batch.numpy()  # Convert to NumPy array if it's a tensor
#         for tokens in token_batch:
#             for i in range(len(tokens) - 1):
#                 bigram = (tokens[i], tokens[i + 1])
#                 bigram_counts[bigram] += 1
#                 unigram_counts[tokens[i]] += 1
#             unigram_counts[tokens[-1]] += 1  # Last token in the sequence
#     return bigram_counts, unigram_counts
# # Training and Evaluation
# bigram_counts, unigram_counts = compute_ngram_statistics(train_loader)
# brown_clustering = BrownClustering(bigram_counts, unigram_counts, num_clusters=50)
# word_to_cluster = brown_clustering.run()
#
# brown_predictor = BrownClusteringPredictor(word_to_cluster)
# brown_predictor.train(train_loader)
#
# # Evaluation logic
# brown_preds, brown_labels = [], []
# for token_ids, tag_ids in val_loader:
#     preds = brown_predictor.predict(token_ids.numpy())
#     brown_preds.append(preds)
#     brown_labels.append(tag_ids.numpy())


# brown_precision, brown_recall, brown_f1 = evaluate_hmm(brown_preds, brown_labels)
# print(f"[  Brown Clustering ] Precision: {brown_precision:.3f} \t Recall: {brown_recall:.3f} \t F1: {brown_f1:.3f}")


#######################
# Decision Trees
#######################
def decision_tree(train_dataset, val_dataset, feature_mapping, label_map):
    class DecisionTreeNER:
        def __init__(self):
            self.model = DecisionTreeClassifier()

        def train(self, X, y):
            self.model.fit(X, y)

        def predict(self, X):
            return self.model.predict(X)

    # Prepare training and validation data
    X_train, y_train = prepare_data_for_decision_tree(train_dataset, feature_mapping)
    X_val, y_val = prepare_data_for_decision_tree(val_dataset, feature_mapping)

    dt_ner = DecisionTreeNER()
    dt_ner.train(X_train, y_train)
    y_pred = dt_ner.predict(X_val)

    # Evaluation
    ignore_label = label_map["O"]
    paired = [(true, pred) for true, pred in zip(y_val, y_pred) if true != ignore_label or pred != ignore_label]
    y_val = [true for true, pred in paired]
    y_pred = [pred for true, pred in paired]

    precision = precision_score(y_val, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_val, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='micro', zero_division=0)

    print(f"[Decision Tree] Precision: {precision:.3f} \t Recall: {recall:.3f} \t F1: {f1:.3f}")
    return precision, recall, f1


############################
# DistilBERT NER
############################
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')


def distilbert_ner(tokenized_datasets, label_map):
    """Train and evaluate DistilBERT for NER."""
    id2label = {v: k for k, v in label_map.items()}

    # Initialize model
    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-multilingual-cased',
        num_labels=len(label_map)
    )
    model.config.id2label = id2label
    model.config.label2id = label_map

    # Freeze all layers except classifier head
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=50
    )

    # Data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Define metrics
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = [
            [id2label[label] for label in label_row if label != -100]
            for label_row in labels
        ]
        true_predictions = [
            [id2label[pred] for pred, label in zip(pred_row, label_row) if label != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]

        metric = load_metric("seqeval")
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
        }

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()

    # Extract metrics
    precision = eval_results["eval_precision"]
    recall = eval_results["eval_recall"]
    f1 = eval_results["eval_f1"]

    # Print metrics in existing format
    print(f"[DistilBERT NER] Precision: {precision:.3f} \t Recall: {recall:.3f} \t F1: {f1:.3f}")
    return precision, recall, f1


#############################################
# BROWN CLUSTERING
#############################################

class SimplifiedBrownClustering:
    def __init__(self, unigram_counts, num_clusters):
        self.unigram_counts = unigram_counts
        self.num_clusters = num_clusters
        self.word_to_cluster = {}

    def run(self):
        # Simple clustering based on word frequency
        sorted_words = sorted(self.unigram_counts.items(), key=lambda x: x[1], reverse=True)
        words = [word for word, _ in sorted_words]

        # Divide words into clusters
        cluster_size = max(1, len(words) // self.num_clusters)
        for i in range(self.num_clusters):
            start = i * cluster_size
            end = start + cluster_size
            for word in words[start:end]:
                self.word_to_cluster[word] = i

        return self.word_to_cluster

class FastBrownClusteringPredictor:
    def __init__(self, word_to_cluster):
        self.word_to_cluster = word_to_cluster
        self.cluster_tag_map = {}

    def train(self, train_data):
        # Precompute most frequent tag for each cluster
        cluster_tag_counts = defaultdict(Counter)
        for example in train_data:
            tokens = example['tokens']
            tags = example['ner_tags']
            for token, tag in zip(tokens, tags):
                cluster_id = self.word_to_cluster.get(token, 0)
                cluster_tag_counts[cluster_id][tag] += 1

        # Get most common tag for each cluster
        self.cluster_tag_map = {
            cluster: counts.most_common(1)[0][0]
            for cluster, counts in cluster_tag_counts.items()
        }

    def predict(self, tokens):
        return [
            self.cluster_tag_map.get(self.word_to_cluster.get(token, 0), 0)
            for token in tokens
        ]

def brown_clustering_ner(multilingual_dataset, num_clusters, label_map):
    """Train and evaluate Brown Clustering NER."""
    # Compute unigram counts
    unigram_counts = compute_unigram_counts(multilingual_dataset)

    # Run Brown Clustering
    brown_clustering = SimplifiedBrownClustering(unigram_counts, num_clusters)
    word_to_cluster = brown_clustering.run()

    # Train predictor
    predictor = FastBrownClusteringPredictor(word_to_cluster)
    predictor.train(multilingual_dataset['train'])

    # Evaluate
    true_tags = []
    predicted_tags = []
    for example in multilingual_dataset['test']:
        tokens = example['tokens']
        true_tags.extend(example['ner_tags'])
        predicted_tags.extend(predictor.predict(tokens))

    # Compute metrics
    report = classification_report(
        true_tags, predicted_tags, output_dict=True, zero_division=0
    )

    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    print(f"[Brown Clustering] Precision: {precision:.3f} \t Recall: {recall:.3f} \t F1: {f1:.3f}")
    return precision, recall, f1
