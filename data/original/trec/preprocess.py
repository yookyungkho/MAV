import json
import pandas as pd
from pandas import DataFrame
""" 
Codes in this file are adapted from LM-BFF(https://github.com/princeton-nlp/LM-BFF/tree/main/data)
TREC Data can be downloaded via below codes
```
wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar xvf datasets.tar
```
"""

ftrain = open('TREC.train.all')
train = []
for line in ftrain:
    line = line.strip()
    label = int(line[0])
    text = line[2:]
    train.append([label, text])
train = DataFrame(train)
train.to_csv('train.csv', header=False, index=False)

ftest = open('TREC.test.all')
test = []
for line in ftest:
    line = line.strip()
    label = int(line[0])
    text = line[2:]
    test.append([label, text])
test = DataFrame(test)
test.to_csv('test.csv', header=False, index=False)