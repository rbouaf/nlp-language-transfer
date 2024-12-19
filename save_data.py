from datasets import load_dataset

#internal import
from langs import langs

# Load the dataset for each language from disk
for lang in langs:
    dataset = load_dataset('unimelb-nlp/wikiann', lang)
    dataset.save_to_disk(f"./processed_wikiann_{lang}")