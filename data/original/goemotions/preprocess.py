import pandas as pd
from tqdm import tqdm

def process_goemotions(dataset_path, over60_class = None, new_label_map=None):
    dataset = pd.read_csv(dataset_path, sep="\t", header=None)
    dataset.columns = ["text", "label", "id"]
    dataset["is_multi_label"] = dataset.apply(lambda x: len(x["label"].split(",")) > 1, axis=1)
    dataset = dataset[dataset["is_multi_label"] == False][["label", "text"]]
    dataset.reset_index(drop=True, inplace=True)
    if new_label_map is None:
        over60_class = dataset.label.value_counts()
        over60_class = {_class for _class, _counts in over60_class.items() if _counts > 60}
    dataset = dataset[dataset.label.isin(over60_class)]
    new_label_map = {fine: new_lab for new_lab, fine in enumerate(set(dataset.label))}
    dataset["label"] = dataset["label"].apply(lambda label: new_label_map[label])
    dataset.reset_index(drop=True, inplace=True)
    return dataset, over60_class, new_label_map 

train_original, over60_class, new_label_map = process_goemotions("original/train.tsv")
test_original, _, _ = process_goemotions("original/test.tsv", over60_class = over60_class, new_label_map=new_label_map)
print(new_label_map)

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