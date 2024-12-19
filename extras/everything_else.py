from langs import lang_combos, label_map
from preprocessing import vocab_and_datasets, preprocess
from preprocessing import number_of_tags, id2label,build_feature_mapping
from models import hmm, lstmcrf, bootstrapping,decision_tree,brown_clustering_ner
import time


for combo in lang_combos:
    print(combo)
    print("Preprocessing for ",combo," starting")
    pp_start = time.time()
    word2id,vocab_size,multilingual_dataset = vocab_and_datasets(combo)
    train_data,val_data,test_data,train_loader,val_loader,test_loader = preprocess(multilingual_dataset,word2id)
    pp_end = time.time()
    print("Preprocessing for ",combo," took ", pp_end - pp_start, " seconds")

    print("HMM for ", combo, " starting")
    hmm_start = time.time()
    hmm_precision, hmm_recall, hmm_f1 = hmm(vocab_size,train_loader, val_loader)
    hmm_end = time.time()
    print("HMM for ", combo, " took ", hmm_end - hmm_start, " seconds")

    print("LSTM-CRF for ", combo, " starting")
    lstmcrf_start = time.time()
    lstmcrf_precision,lstmcrf_recall,lstmcrf_f1 = lstmcrf(vocab_size,train_loader,val_loader,word2id,number_of_tags)
    lstmcrf_end = time.time()
    print("LSTM-CRF for ", combo, " took ", lstmcrf_end - lstmcrf_start, " seconds")

    print("Bootstrapping for ", combo, " starting")
    bootstrap_start = time.time()
    bootstrap_precision, bootstrap_recall, bootstrap_f1 = bootstrapping(id2label,val_loader)
    bootstrap_end = time.time()
    print("Bootstrapping for ", combo, " took ", bootstrap_end - bootstrap_start, " seconds")

    # Decision Tree requires feature mapping
    print(f"Building feature mapping for Decision Tree (if required) for {combo}")
    feature_mapping = build_feature_mapping(multilingual_dataset)

    print(f"Running Decision Tree NER for {combo}")
    dt_start = time.time()
    dt_precision, dt_recall, dt_f1 = decision_tree(
        multilingual_dataset['train'], multilingual_dataset['validation'], feature_mapping, label_map
    )
    dt_end = time.time()
    print(f"Decision Tree for {combo} took {dt_end - dt_start:.2f} seconds")
    # Run Brown Clustering with fixed cluster sizes
    cluster_sizes = [5, 10, 20, 30, 40, 50]
    for num_clusters in cluster_sizes:
        print(f"Running Brown Clustering with {num_clusters} clusters for {combo}")
        brown_start = time.time()
        brown_precision, brown_recall, brown_f1 = brown_clustering_ner(
            multilingual_dataset, num_clusters, label_map
        )
        brown_end = time.time()
        print(f"Brown Clustering with {num_clusters} clusters for {combo} took {brown_end - brown_start:.2f} seconds")