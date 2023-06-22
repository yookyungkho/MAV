"""
The data source is https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?resource=download&select=train.csv \
It is saved in this directory by **'train_original'** and **'test_original.csv'**
"""

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os

def preprocess(text:str) -> str:
    text = text.replace("\\n", " ").replace("\\", " ").strip("")
    " " if text == "" else text
    return text

train_original = pd.read_csv("train_original.csv")
test_original = pd.read_csv("test_original.csv")

print("----Processing Train----")
train = []
for idx in tqdm(range(train_original.shape[0])) :
    line = train_original.loc[idx].to_list()
    label, (title, body) = int(line[0]-1), line[1:] # The original first class was mapped to 1 not 0
    if len(body) > 4000 : # limit length of input
        continue
    line = [label] + [title] + [body]
    train.append(line)

train = pd.DataFrame(train)
train.to_csv('train.csv', header=False, index=False)

print("----Processing Test----")
test = []
for idx in tqdm(range(test_original.shape[0])) :
    line = test_original.loc[idx].to_list()
    label, (title, body) = int(line[0]-1), line[1:] # The original first class was mapped to 1 not 0
    line = [label] + [title] + [body]
    test.append(line)
    
test = pd.DataFrame(test)
test.to_csv('test.csv', header=False, index=False)