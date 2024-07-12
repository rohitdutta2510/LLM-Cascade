import json
# import html
# import tokenize
import torch
# from math import ceil
import os
# import re
import sys
import gc
import pandas as pd
from tqdm import tqdm
from time import time, sleep
import torch.utils.data as data_utils
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
# from ptflops import get_model_complexity_info
# from fvcore.nn import FlopCountAnalysis
# from experiment_impact_tracker.compute_tracker import ImpactTracker
from carbontracker.tracker import CarbonTracker
# from carbontracker import parser
from argparse import ArgumentParser


def argparse():
    parser = ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("--out_dir", type=str, default= '../../temp_outs/temp_outs_default/')
    parser.add_argument("--finetune_path", type=str, default='')
    parser.add_argument("--bs", default=16, type=int)
    parser.add_argument("--max_gen_tokens", default = 10, type=int)
    
    
    return parser.parse_args()


############################################################################################################
# %% SET UP PARAMS

torch.cuda.empty_cache()

args = argparse()

MODEL_PATH = args.model_path
DATA_PATH = args.data_path
OUT_DIR_BASE = args.out_dir
FT_PATH = args.finetune_path
BATCH_SIZE = args.bs
MAXGENTOKENS = args.max_gen_tokens


DEVICE = 'cuda:0'
TRACK_GPU = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else DEVICE[-1]

print("\n")
print("#"*50)
print("#"*50)
print(TRACK_GPU, args)
print("#"*50)

MODEL_NAME = MODEL_PATH[MODEL_PATH.rindex('/')+1:]
DATA_NAME = DATA_PATH[DATA_PATH.rindex('/')+1:DATA_PATH.rindex('.')]
OUT_DIR = os.path.join(OUT_DIR_BASE, '%s_%s'%(DATA_NAME, MODEL_NAME))


if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
else:
    os.system("rm " + os.path.join(OUT_DIR, "emissions.csv"))
    os.system("rm -r " + os.path.join(OUT_DIR, "carbon_tracker/"))


############################################################################################################
# %% LOAD TOKENIZER AND MODEL

if 'Phi-3' in MODEL_PATH:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if 'flan-t5' in MODEL_PATH:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, device_map = DEVICE, torch_dtype=torch.float16)
elif 'Phi-3-medium-4k-instruct' in MODEL_PATH:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=quantization_config, device_map = DEVICE, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="flash_attention_2")
elif 'Phi-3' in MODEL_PATH:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map = DEVICE, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map = DEVICE, torch_dtype=torch.bfloat16)
    # model.bfloat16()
    # if not tokenizer.pad_token_id:
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    if FT_PATH:
        model = PeftModel.from_pretrained(model, FT_PATH)
        model = model.merge_and_unload()
        print("LOADED PEFT PARAMS")
        # tokenizer.padding_side = "right"

    # model.resize_token_embeddings(len(tokenizer))


############################################################################################################
# %% LOAD DATA
rawdata = pd.read_csv(DATA_PATH).prompt_text
rawdata = rawdata.sort_values(key = lambda col: col.apply(len))
rawdata = rawdata.values.tolist()
data_loader = data_utils.DataLoader(rawdata, batch_size=BATCH_SIZE)



results = []
timestamps = []

# print("#" * 50 + "\nstarting\n" + "#" * 50 , flush = True)



############################################################################################################
# %% TOKENIZE DATA AND RUN LLM 

try:
    for idx, batch in enumerate(tqdm(data_loader, ncols = 50)):

        # tracker1 = CarbonTracker(epochs=1, update_interval=1, devices_by_pid=True, verbose=False, log_file_prefix='%d_'%idx,
        #                      log_dir=os.path.join(OUT_DIR, "carbon_tracker"), epochs_before_pred=-1, monitor_epochs=-1)
        
        with OfflineEmissionsTracker(project_name="%s_%s_%d"%(DATA_NAME, MODEL_NAME, idx), experiment_id=idx, country_iso_code="IND", log_level="error",
                                     tracking_mode="process", output_dir=OUT_DIR, measure_power_secs=1, gpu_ids=TRACK_GPU) as tracker2:
            st = time()
            # tracker1.epoch_start()

            batchdata = tokenizer(batch, return_tensors="pt", padding = True, truncation =  True)
            inp = batchdata.input_ids.to(DEVICE)
            attn = batchdata.attention_mask.to(DEVICE)
            
            outputs = model.generate(inp, attention_mask=attn, max_new_tokens=MAXGENTOKENS, pad_token_id=tokenizer.pad_token_id)
            
            # tracker1.epoch_end()
            end = time()
        
        results.extend(outputs)
        timestamps.append((st,end))

        # tracker1.stop()
        gc.collect()
        torch.cuda.empty_cache()

except Exception as e:
    print("ERROR:", e)
    tracker2.stop()
    # tracker1.stop()
    raise e


############################################################################################################
# %% DUMP RESULTS
with open(os.path.join(OUT_DIR, "output.json"), 'w', encoding ='utf8') as fo:
    json.dump([tokenizer.batch_decode(results, skip_special_tokens=True), timestamps], fo, indent = 4, ensure_ascii=False)


torch.cuda.empty_cache()


