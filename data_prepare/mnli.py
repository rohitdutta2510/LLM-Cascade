import os
import html
import re
import random
from datasets import load_dataset
import pandas as pd


# DATADIR = '../../old/wnli/'
OUTPATH = '../../../LLM-Cascade/datasets/mnli_test_1k.csv'
NUM_DATA = 1024



# with open(os.path.join(DATADIR, 'instructions.txt')) as fp:
    # prompt1 = fp.read().strip()

prompt1 = 'Select the stance of the premise towards the hypothesis: Entailment (0), Neutral (1), Contradiction (2)'
prompt2 = "\nPremise: "
prompt3 = "\nHypothesis: "
prompt4 = "\n Answer: "



data = load_dataset("nyu-mll/glue", 'mnli', split="validation_matched")
data = pd.DataFrame.from_records(data)
data = data.sample(NUM_DATA)


prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + item['premise'] + prompt3 + item['hypothesis'] + prompt4
    prompt_texts.append(prompt)
    

data = data.drop(columns=['premise', 'hypothesis'])
data["prompt_text"] = prompt_texts

data.to_csv(OUTPATH, index=False)