import time
from langs import langs, label_map
from preprocessing import vocab_and_datasets, preprocess, preprocess_for_distilbert, build_feature_mapping
from models import hmm, lstmcrf, bootstrapping, brown_clustering_ner, distilbert_ner, decision_tree
from torch.utils.data import DataLoader, Dataset

# Define high-resource and low-resource language groups
high_resource = ['en', 'fr', 'zh']
low_resource = ['ar', 'fa', 'sw', 'fi']

# Sequential Transfer Learning
print("Sequential Transfer Learning Experiment")

# Pre-train on high-resource languages
print(f"Pretraining on high-resource languages: {high_resource}")
word2id, vocab_size, multilingual_dataset = vocab_and_datasets(high_resource)
train_data, val_data, test_data, train_loader, val_loader, test_loader = preprocess(multilingual_dataset, word2id)

# HMM Pre-training
print("Pretraining HMM on high-resource languages")
hmm_precision, hmm_recall, hmm_f1 = hmm(vocab_size, train_loader, val_loader)
print(f"HMM Pretraining | Precision: {hmm_precision:.3f}, Recall: {hmm_recall:.3f}, F1: {hmm_f1:.3f}")

# LSTM-CRF Pre-training
print("Pretraining LSTM-CRF on high-resource languages")
lstmcrf_model, pretrained_weights = lstmcrf(
    vocab_size=vocab_size,
    train_loader=train_loader,
    val_loader=val_loader,
    word2id=word2id,
    number_of_tags=len(label_map),
    return_pretrained=True
)

# Bootstrapping Pre-training
print("Pretraining Bootstrapping on high-resource languages")
bootstrap_precision, bootstrap_recall, bootstrap_f1 = bootstrapping(label_map, val_loader)
print(
    f"Bootstrapping Pretraining | Precision: {bootstrap_precision:.3f}, Recall: {bootstrap_recall:.3f}, F1: {bootstrap_f1:.3f}")

# Fine-tune on each low-resource language
for lang in low_resource:
    print(f"Fine-tuning on low-resource language: {lang}")
    word2id, vocab_size, multilingual_dataset = vocab_and_datasets([lang])
    train_data, val_data, test_data, train_loader, val_loader, test_loader = preprocess(multilingual_dataset, word2id)

    # HMM Fine-tuning
    hmm_precision, hmm_recall, hmm_f1 = hmm(vocab_size, train_loader, val_loader)
    print(f"HMM for {lang} | Precision: {hmm_precision:.3f}, Recall: {hmm_recall:.3f}, F1: {hmm_f1:.3f}")

    # LSTM-CRF Fine-tuning
    lstmcrf_precision, lstmcrf_recall, lstmcrf_f1 = lstmcrf(
        vocab_size=vocab_size,
        train_loader=train_loader,
        val_loader=val_loader,
        word2id=word2id,
        number_of_tags=len(label_map),
        pretrained_weights=pretrained_weights
    )
    print(
        f"LSTM-CRF for {lang} | Precision: {lstmcrf_precision:.3f}, Recall: {lstmcrf_recall:.3f}, F1: {lstmcrf_f1:.3f}")

    # Bootstrapping Fine-tuning
    bootstrap_precision, bootstrap_recall, bootstrap_f1 = bootstrapping(label_map, val_loader)
    print(
        f"Bootstrapping for {lang} | Precision: {bootstrap_precision:.3f}, Recall: {bootstrap_recall:.3f}, F1: {bootstrap_f1:.3f}")

    # Decision Tree
    feature_mapping = build_feature_mapping(multilingual_dataset)
    dt_precision, dt_recall, dt_f1 = decision_tree(
        multilingual_dataset['train'],
        multilingual_dataset['validation'],
        feature_mapping,
        label_map
    )
    print(f"Decision Tree for {lang} | Precision: {dt_precision:.3f}, Recall: {dt_recall:.3f}, F1: {dt_f1:.3f}")

    # Brown Clustering
    cluster_sizes = [5, 10, 20]
    for num_clusters in cluster_sizes:
        brown_precision, brown_recall, brown_f1 = brown_clustering_ner(
            multilingual_dataset, num_clusters, label_map
        )
        print(
            f"Brown Clustering ({num_clusters} clusters) for {lang} | Precision: {brown_precision:.3f}, Recall: {brown_recall:.3f}, F1: {brown_f1:.3f}")

# Zero-shot Transfer Learning
print("\nZero-shot Transfer Learning Experiment")
print(f"Training on high-resource languages: {high_resource} and testing directly on low-resource languages.")

# Pre-train DistilBERT on high-resource languages
word2id, vocab_size, multilingual_dataset = vocab_and_datasets(high_resource)
tokenized_datasets = preprocess_for_distilbert(multilingual_dataset)

distilbert_model = distilbert_ner(tokenized_datasets, label_map, return_pretrained=True)

# Evaluate directly on low-resource languages
for lang in low_resource:
    print(f"Evaluating DistilBERT zero-shot on low-resource language: {lang}")
    word2id, vocab_size, multilingual_dataset = vocab_and_datasets([lang])
    tokenized_datasets = preprocess_for_distilbert(multilingual_dataset)

    distilbert_precision, distilbert_recall, distilbert_f1 = distilbert_ner(
        tokenized_datasets=tokenized_datasets,
        label_map=label_map,
        pretrained_weights=distilbert_model
    )
    print(
        f"DistilBERT for {lang} | Precision: {distilbert_precision:.3f}, Recall: {distilbert_recall:.3f}, F1: {distilbert_f1:.3f}")

# Few-shot Learning
print("\nFew-shot Learning Experiment")
percentages = [5, 10, 20]

for percentage in percentages:
    for lang in low_resource:
        print(f"Fine-tuning on {percentage}% of {lang}'s data")
        word2id, vocab_size, multilingual_dataset = vocab_and_datasets([lang])

        # Reduce training set
        train_data = multilingual_dataset['train'].select(
            range(int(len(multilingual_dataset['train']) * (percentage / 100))))
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # LSTM-CRF
        lstmcrf_precision, lstmcrf_recall, lstmcrf_f1 = lstmcrf(
            vocab_size=vocab_size,
            train_loader=train_loader,
            val_loader=val_loader,
            word2id=word2id,
            number_of_tags=len(label_map),
            pretrained_weights=pretrained_weights
        )
        print(
            f"{percentage}% Few-shot LSTM-CRF for {lang} | Precision: {lstmcrf_precision:.3f}, Recall: {lstmcrf_recall:.3f}, F1: {lstmcrf_f1:.3f}")
