import os
import html
import re
import random
from datasets import load_dataset
import pandas as pd


# DATADIR = '../../old/cnn_dm/'
OUTPATH = '../datasets/cnn_dm_test_1k.csv'
NUM_DATA = 1024



# with open(os.path.join(DATADIR, 'instructions.txt')) as fp:
#     prompt1 = fp.read().strip()
prompt1 = "Summarize the following news article in about 50 words:"
prompt2 = "\nInput: "
prompt3 = "\nOutput: "



data = load_dataset("cnn_dailymail", '3.0.0', split = "validation")
data = pd.DataFrame.from_records(data)
data = data.sample(NUM_DATA)


prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + item['article'] + prompt3
    
    prompt_texts.append(prompt)
    

data = data.drop(columns=['article'])
data["prompt_text"] = prompt_texts

data.to_csv(OUTPATH, index=False)