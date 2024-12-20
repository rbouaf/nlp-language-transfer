# List of languages we will test'en','fr', 'zh', 'ar', 'fa','sw', 'fi'
langs = ['en','fr', 'zh', 'ar', 'fa','sw', 'fi']
# Label mapping (given by WikiAnn)
label_map = {"O": 0,"B-PER": 1,"I-PER": 2,"B-ORG": 3,"I-ORG": 4,"B-LOC": 5,"I-LOC": 6}

from itertools import combinations


# Generate every combination of lengths 1 through n-1
all_combinations = []
n = len(langs)
for r in range(1, n):
    all_combinations.extend(combinations(langs, r))
lang_combos = list(all_combinations)
lang_combos.append(langs)


langs1 = []
for i in langs:
    langs1.append(list[i])
