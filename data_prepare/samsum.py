import os
import html
import re
import random
from datasets import load_dataset
import pandas as pd


# DATADIR = '../../old/wnli/'
TRAIN_OUTPATH = '../datasets/samsum_train_1k.csv'
TEST_OUTPATH = '../datasets/samsum_test_1k.csv'
NUM_DATA = 2048

split_size = 1024

# with open(os.path.join(DATADIR, 'instructions.txt')) as fp:
#     prompt1 = fp.read().strip()
prompt1 = 'Summarize the given dialogue. Keep the summary short.'
prompt2 = "\ndialogue: "
prompt3 = "\nsummary: "


data = load_dataset("samsum", split="train", trust_remote_code=True)
data = pd.DataFrame.from_records(data)
data = data.sample(NUM_DATA)


prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + item['dialogue'] + prompt3 
    prompt_texts.append(prompt)
    

data = data.drop(columns=['dialogue'])
data["prompt_text"] = prompt_texts

train_df = data.iloc[:split_size]
test_df = data.iloc[split_size:]

train_df.to_csv(TRAIN_OUTPATH, index=False)
test_df.to_csv(TEST_OUTPATH, index=False)
