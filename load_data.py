from datasets import load_from_disk
from langs import langs as languages
def load_data(langs):
    data_paths = [f"./processed_wikiann_{lang}" for lang in langs]
    datasets = {}
    for lang, path in zip(langs, data_paths):
        datasets[lang] = load_from_disk(path)
        print(f"Loaded {lang} dataset")
    return datasets

def load_all_data():
    data_paths = [f"./processed_wikiann_{lang}" for lang in languages]
    datasets = {}
    for lang, path in zip(languages, data_paths):
        datasets[lang] = load_from_disk(path)
        print(f"Loaded {lang} dataset")
    return datasets
