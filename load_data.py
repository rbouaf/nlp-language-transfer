from datasets import load_dataset, load_from_disk

# List of languages we will test
langs = ['en','fr', 'zh', 'ar', 'fa','sw', 'fi']
# Label mapping (given by WikiAnn)
label_map = {"O": 0,"B-PER": 1,"I-PER": 2,"B-ORG": 3,"I-ORG": 4,"B-LOC": 5,"I-LOC": 6}

# Load the dataset for each language
for lang in langs:
    dataset = load_dataset('unimelb-nlp/wikiann', lang)
    dataset.save_to_disk(f"./processed_wikiann_{lang}")

def load_data():
    data_paths = [f"./processed_wikiann_{lang}" for lang in langs]
    datasets = {}
    for lang, path in zip(langs, data_paths):
        datasets[lang] = load_from_disk(path)
        print(f"Loaded {lang} dataset")
    return datasets
