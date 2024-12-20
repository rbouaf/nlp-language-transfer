import time
from langs import langs, label_map
from preprocessing import vocab_and_datasets, preprocess, preprocess_for_distilbert, build_feature_mapping
from models import hmm, lstmcrf, bootstrapping, brown_clustering_ner, distilbert_ner, decision_tree
from torch.utils.data import DataLoader

# Define language groups
high_resource = ['en', 'fr', 'zh']
low_resource = ['ar', 'fa', 'sw', 'fi']
id2label = {v: k for k, v in label_map.items()}

# Pre-train on High-Resource Languages
print("\nPretraining on High-Resource Languages")
word2id, vocab_size, multilingual_dataset = vocab_and_datasets(high_resource)
train_data, val_data, test_data, train_loader, val_loader, test_loader = preprocess(multilingual_dataset, word2id)


# DistilBERT Pre-training
print("Pretraining DistilBERT on high-resource languages")
tokenized_datasets = preprocess_for_distilbert(multilingual_dataset)
distilbert_model = distilbert_ner(
    tokenized_datasets=tokenized_datasets,
    label_map=label_map
)

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


        # DistilBERT
        tokenized_datasets = preprocess_for_distilbert(multilingual_dataset)
        distilbert_precision, distilbert_recall, distilbert_f1 = distilbert_ner(
            tokenized_datasets=tokenized_datasets,
            label_map=label_map
        )
        print(f"{percentage}% Few-shot DistilBERT for {lang} | Precision: {distilbert_precision:.3f}, Recall: {distilbert_recall:.3f}, F1: {distilbert_f1:.3f}")
