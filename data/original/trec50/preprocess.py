from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def process_trec50(data : pd.DataFrame, over60_class : set, new_label_map : dict = None) :
    data["is_class_over60"] = data.apply(lambda line : line["fine_label"] in over60_class, axis = 1)
    data = data[data.is_class_over60][["fine_label", "text"]]
    data.reset_index(drop=True, inplace=True)
    if new_label_map is None :
        new_label_map = {fine : new_lab for new_lab, fine in enumerate(set(data.fine_label))}
    data["fine_label"] = data["fine_label"].apply(lambda label : new_label_map[label])
    return data, new_label_map

dataset = load_dataset("trec")

train_original = dataset["train"].to_pandas()
test_original = dataset["test"].to_pandas()

over60_class = train_original.fine_label.value_counts()
over60_class = {_class for _class, _counts in over60_class.items() if _counts > 60}

train_original, new_label_map = process_trec50(train_original, over60_class)
test_original, _ = process_trec50(test_original, over60_class, new_label_map)

print("----Processing Train----")
train = []
for idx in tqdm(range(train_original.shape[0])) :
    line = train_original.loc[idx].to_list()
    label, inputs = int(line[0]), line[1:]
    inputs = [sentence.replace("\\n", "") for sentence in inputs]
    line = [label] + inputs
    train.append(line)

train = pd.DataFrame(train)
train.to_csv('train.csv', header=False, index=False)

print("----Processing Test----")
test = []
for idx in tqdm(range(test_original.shape[0])) :
    line = test_original.loc[idx].to_list()
    label, inputs = int(line[0]), line[1:]
    inputs = [sentence.replace("\\n", "") for sentence in inputs]
    line = [label] + inputs
    test.append(line)
    
test = pd.DataFrame(test)
test.to_csv('test.csv', header=False, index=False)