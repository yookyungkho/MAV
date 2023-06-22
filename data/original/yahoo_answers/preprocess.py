from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def preprocess(text:str) -> str:
    text = text.replace("\\n", " ").replace("\\", " ").strip("")
    " " if text == "" else text
    return text

dataset = load_dataset("yahoo_answers_topics")

train_original = dataset["train"].to_pandas()
test_original = dataset["test"].to_pandas()

print("----Processing Train----")
train = []
for idx in tqdm(range(train_original.shape[0])) :
    line = train_original.loc[idx].to_list()
    label, inputs = int(line[1]), line[2:5] # (question_title, question_body, answer)
    inputs = [preprocess(sent) for sent in inputs]
    inputs = " ".join(inputs)
     
    if len(inputs) > 5000 :
        continue

    line = [label] + [inputs]
    train.append(line)
    # question = inputs[0] + " " + inputs[1]
    # answer = inputs[2]

    # if (len(question) + len(answer)) > 5000 : # limit length of input
    #     continue

    # line = [label] + [question] + [answer]
    # train.append(line)

train = pd.DataFrame(train)
train.to_csv('train.csv', header=False, index=False)

print("----Processing Test----")
test = []
for idx in tqdm(range(test_original.shape[0])) :
    line = test_original.loc[idx].to_list()
    label, inputs = int(line[1]), line[2:5] # (question_title, question_body, answer)
    inputs = [preprocess(sent) for sent in inputs]
    inputs = " ".join(inputs)
    line = [label] + [inputs]
    test.append(line)
    
test = pd.DataFrame(test)
test.to_csv('test.csv', header=False, index=False)
