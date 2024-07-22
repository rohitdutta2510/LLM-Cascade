import os
import html
import re
import random
from datasets import load_dataset
import pandas as pd


OUTPATH = '../../datasets/datasets_llm_cascade/boolq_test_1k.csv'
NUM_DATA = 1024



# with open(os.path.join(DATADIR, 'instructions.txt')) as fp:
#     prompt1 = fp.read().strip()
prompt1 = 'Read the passage and answer the question with True or False.'
prompt2 = "\nquestion: "
prompt3 = "\npassage: "
prompt4 = "\nAnswer: "



data = load_dataset("super_glue", 'boolq', split="validation")
data = pd.DataFrame.from_records(data)
data = data.sample(NUM_DATA)


prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + item['question']\
     + prompt3 + item['passage'] + prompt4
    prompt_texts.append(prompt)
    

data = data.drop(columns=['question','passage'])
data["prompt_text"] = prompt_texts

data.to_csv(OUTPATH, index=False)