from langs import lang_combos,label_map
from preprocessing import vocab_and_datasets, preprocess
from preprocessing import preprocess_for_distilbert
from models import distilbert_ner
import time


for combo in lang_combos:
    print(combo)
    print("Preprocessing for ",combo," starting")
    pp_start = time.time()
    word2id,vocab_size,multilingual_dataset = vocab_and_datasets(combo)
    train_data,val_data,test_data,train_loader,val_loader,test_loader = preprocess(multilingual_dataset,word2id)
    pp_end = time.time()
    print("Preprocessing for ",combo," took ", pp_end - pp_start, " seconds")

    print(f"Preprocessing for DistilBERT for {combo}")
    tokenized_datasets = preprocess_for_distilbert(multilingual_dataset)

    print(f"Running DistilBERT NER for {combo}")
    distilbert_start = time.time()
    distilbert_precision, distilbert_recall, distilbert_f1 = distilbert_ner(
        tokenized_datasets, label_map
    )
    distilbert_end = time.time()
    print(f"DistilBERT for {combo} took {distilbert_end - distilbert_start:.2f} seconds")