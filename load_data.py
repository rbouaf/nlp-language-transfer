from datasets import load_dataset, DatasetDict

# List of desired languages
languages = ['en', 'fr', 'zh', 'ar', 'fa', 'sw', 'fi']

# Load the dataset for each language
for lang in languages:
    dataset = load_dataset('unimelb-nlp/wikiann', lang)
    print(lang)
    print(dataset['train'][0])
    # Access the dataset splits
    dataset.save_to_disk(f"./processed_wikiann_{lang}")