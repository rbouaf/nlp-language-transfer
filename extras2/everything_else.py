import time
from langs import langs, label_map
from preprocessing import vocab_and_datasets, preprocess, build_feature_mapping
from models import hmm, lstmcrf, brown_clustering_ner, decision_tree
from torch.utils.data import DataLoader

# Define language groups
high_resource = ['en', 'fr', 'zh']
low_resource = ['ar', 'fa', 'sw', 'fi']
id2label = {v: k for k, v in label_map.items()}

# Pre-train on High-Resource Languages
print("\nPretraining on High-Resource Languages")
word2id, vocab_size, multilingual_dataset = vocab_and_datasets(high_resource)
train_data, val_data, test_data, train_loader, val_loader, test_loader = preprocess(multilingual_dataset, word2id)

# HMM Pre-training
print("Pretraining HMM on high-resource languages")
hmm_precision, hmm_recall, hmm_f1 = hmm(vocab_size, train_loader, val_loader)
print(f"HMM Pretraining | Precision: {hmm_precision:.3f}, Recall: {hmm_recall:.3f}, F1: {hmm_f1:.3f}")

# LSTM-CRF Pre-training
print("Pretraining LSTM-CRF on high-resource languages")
lstmcrf_model = lstmcrf(
    vocab_size=vocab_size,
    train_loader=train_loader,
    val_loader=val_loader,
    word2id=word2id,
    number_of_tags=len(label_map)
)

# Decision Tree Pre-training
print("Pretraining Decision Tree on high-resource languages")
feature_mapping = build_feature_mapping(multilingual_dataset)
dt_precision, dt_recall, dt_f1 = decision_tree(
    multilingual_dataset['train'],
    multilingual_dataset['validation'],
    feature_mapping,
    label_map
)
print(f"Decision Tree Pretraining | Precision: {dt_precision:.3f}, Recall: {dt_recall:.3f}, F1: {dt_f1:.3f}")

# Brown Clustering Pre-training
print("Pretraining Brown Clustering on high-resource languages")
for num_clusters in [5, 10, 20]:
    brown_precision, brown_recall, brown_f1 = brown_clustering_ner(
        multilingual_dataset, num_clusters, label_map
    )
    print(f"Brown Clustering ({num_clusters} clusters) Pretraining | Precision: {brown_precision:.3f}, Recall: {brown_recall:.3f}, F1: {brown_f1:.3f}")



# Few-shot Learning
print("\nFew-shot Learning Experiment")
percentages = [5, 10, 20]

for percentage in percentages:
    for lang in low_resource:
        print(f"Fine-tuning on {percentage}% of {lang}'s data")
        word2id, vocab_size, multilingual_dataset = vocab_and_datasets([lang])

        # Reduce training set
        train_data = multilingual_dataset['train'].select(range(int(len(multilingual_dataset['train']) * (percentage / 100))))
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        train_data, val_data, test_data, train_loader, val_loader, test_loader = preprocess(multilingual_dataset, word2id)

        # HMM
        hmm_precision, hmm_recall, hmm_f1 = hmm(vocab_size, train_loader, val_loader)
        print(f"{percentage}% Few-shot HMM for {lang} | Precision: {hmm_precision:.3f}, Recall: {hmm_recall:.3f}, F1: {hmm_f1:.3f}")

        # LSTM-CRF
        lstmcrf_precision, lstmcrf_recall, lstmcrf_f1 = lstmcrf(
            vocab_size=vocab_size,
            train_loader=train_loader,
            val_loader=val_loader,
            word2id=word2id,
            number_of_tags=len(label_map)
        )
        print(f"{percentage}% Few-shot LSTM-CRF for {lang} | Precision: {lstmcrf_precision:.3f}, Recall: {lstmcrf_recall:.3f}, F1: {lstmcrf_f1:.3f}")

        # Decision Tree
        feature_mapping = build_feature_mapping(multilingual_dataset)
        dt_precision, dt_recall, dt_f1 = decision_tree(
            multilingual_dataset['train'],
            multilingual_dataset['validation'],
            feature_mapping,
            label_map
        )
        print(f"{percentage}% Few-shot Decision Tree for {lang} | Precision: {dt_precision:.3f}, Recall: {dt_recall:.3f}, F1: {dt_f1:.3f}")

        # Brown Clustering
        for num_clusters in [5, 10, 20]:
            brown_precision, brown_recall, brown_f1 = brown_clustering_ner(
                multilingual_dataset, num_clusters, label_map
            )
            print(f"{percentage}% Few-shot Brown Clustering ({num_clusters} clusters) for {lang} | Precision: {brown_precision:.3f}, Recall: {brown_recall:.3f}, F1: {brown_f1:.3f}")

