import os
import gc
import json
import pandas as pd
import numpy as np
import evaluate
import torch
from tqdm import tqdm
from time import time
from statistics import mean
import torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from transformers import BigBirdTokenizerFast, BigBirdForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from sklearn.metrics import f1_score, precision_score, recall_score

###############################################################################################################

PREDDIR = '../temp_outs/'
TRAIN_DATA_PATH = '../datasets/samsum_train_1k.csv'
TEST_DATA_PATH = '../datasets/samsum_test_1k.csv'
PERF_OUT_DIR = '../performance/'
ENERGY_OUT_DIR_BASE = '../temp_outs/'

DEVICE = 'cuda:0'
TRACK_GPU = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else DEVICE[-1]

################################################################################################################

def get_data_samsum(datadir, preddir, model_name):
    # rouge_score = evaluate.load('rouge')
    bert_score = evaluate.load("bertscore")

    rawdata = pd.read_csv(datadir)
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['summary'].apply(str.lower).values.tolist()

    dn = preddir + '/samsum_train_1k_' + model_name + '/'
        
    try:
        with open(os.path.join(dn, "output.json")) as fp:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                pred_data, ts = json.load(fp)
            
            qa = pred_data.copy() # train copy 
            
            if 'flan' not in model_name:
                pred_data = [pr[len(raw):] for pr, raw in zip(pred_data, rawdata.prompt_text)]
            
            pred_data = map(str.lower, pred_data) 
            pred = list(pred_data)

    except FileNotFoundError:
        print("Skipping", dn)       
    
    # rg = rouge_score.compute(predictions=pred, references=gold)
    bt = bert_score.compute(predictions=pred, references=gold, model_type="distilbert-base-uncased")
    labels = []

    class_cnt = 0
    for sc in bt['f1']:
        if sc > 0.8:
            labels.append(1)
            class_cnt += 1
        else:
            labels.append(0)

    print(f'\nModel: {model_name}, Class 1: {class_cnt}')

    # print(gold[:5])
    # print(pred[:5])
    # print(result[:5])
    # print(qa[:5])

    return {'query_answer' : qa, 'score' : labels}

################################################################################################################

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

################################################################################################################

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

################################################################################################################

def train(train_texts, train_labels, lr, epochs, scorer_path = 'google/bigbird-roberta-base', save_dir = './samsum'):
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
    
    tokenizer = AutoTokenizer.from_pretrained(scorer_path)
    model = AutoModelForSequenceClassification.from_pretrained(scorer_path)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=4096)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=4096)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='../checkpoints',          
        num_train_epochs=epochs,              
        learning_rate=lr,  
        per_device_train_batch_size=4,       
        per_device_eval_batch_size=64,        
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,        
        eval_dataset=val_dataset,            
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
        
################################################################################################################

def build_cascade(llm_chain, cascade_name = 'default', lr = 1e-5, epochs = 10, scorer_path = 'google/bigbird-roberta-base'):

    for model_path in llm_chain:
        model = model_path.split('/')[-1]
        print(f'\nTraining Scorer for {model}\n')

        data = get_data_samsum(TRAIN_DATA_PATH, PREDDIR, model)
        # break
        save_path = os.path.join('../models/samsum', cascade_name, model)
        train(data['query_answer'], data['score'], lr, epochs, scorer_path, save_path)

    log_path = os.path.join('../models/samsum', cascade_name, 'log.csv')
    logs = pd.DataFrame([[lr, epochs, scorer_path]], columns = ["Learning rate", "Epochs", "Scorer"])
    logs.to_csv(log_path, index = False)

    print('\n\nBuilding LLM Cascade successful !!')

################################################################################################################

def get_score(text, scorer_path, data_name, cascade_name, thresh, batch_offset, ENERGY_OUT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(scorer_path)
    model = AutoModelForSequenceClassification.from_pretrained(scorer_path)
    model = model.to(DEVICE)

    test_encodings = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
    test_encodings = {key: value.to(DEVICE) for key, value in test_encodings.items()}
    
    dataset = data_utils.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
    dataloader = data_utils.DataLoader(dataset, batch_size=4)

    all_probs = []
    new_batch_offset = None
    model.eval()
    try:
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, ncols=50)):
                bn = batch_offset + idx
                with OfflineEmissionsTracker(project_name="%s_%s-%s_%d"%(data_name, cascade_name, thresh, bn), experiment_id=bn, country_iso_code="IND", log_level="error",
                                        tracking_mode="process", output_dir=ENERGY_OUT_DIR, measure_power_secs=1, gpu_ids=TRACK_GPU) as tracker2:
                    input_ids, attention_mask = batch
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())
                new_batch_offset = bn

                gc.collect()
                torch.cuda.empty_cache()

    except Exception as e:
        print("ERROR:", e)
        tracker2.stop()
        raise e
    
    new_batch_offset +=1

    return all_probs, new_batch_offset

################################################################################################################

def load_model(model_path):
    # Load Tokenizer
    if 'Phi-3' in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load Model
    if 'flan-t5' in model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map = DEVICE, torch_dtype=torch.float16)
    elif 'Phi-3-medium-4k-instruct' in model_path:
        print("\nLoading 4-bit quantized model\n")
        nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config, device_map = DEVICE, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="flash_attention_2")
    elif 'Phi-3' in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map = DEVICE, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map = DEVICE, torch_dtype=torch.bfloat16)

    # Configure eos and pad tokens
    if 'flan-t5' not in model_path:
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

################################################################################################################

def run_llm(model_path, data_loader, data_name, cascade_name, thresh, batch_offset, ENERGY_OUT_DIR, maxgentokens=50):
    # Load Model and Tokenizer
    model, tokenizer = load_model(model_path)

    results = []
    timestamps = []
    new_batch_offset = None

    try:
        for idx, batch in enumerate(tqdm(data_loader, ncols = 50)):
            bn = batch_offset + idx
            with OfflineEmissionsTracker(project_name="%s_%s-%s_%d"%(data_name, cascade_name, thresh, bn), experiment_id=bn, country_iso_code="IND", log_level="error",
                                     tracking_mode="process", output_dir=ENERGY_OUT_DIR, measure_power_secs=1, gpu_ids=TRACK_GPU) as tracker2:
                st = time()

                batchdata = tokenizer(batch, return_tensors="pt", padding = True, truncation =  True)
                inp = batchdata.input_ids.to(DEVICE)
                attn = batchdata.attention_mask.to(DEVICE)
                
                outputs = model.generate(inp, attention_mask=attn, max_new_tokens=maxgentokens, pad_token_id=tokenizer.pad_token_id)

                end = time()

            results.extend(outputs)
            timestamps.append((st,end))
            new_batch_offset = bn

            gc.collect()
            torch.cuda.empty_cache()

    except Exception as e:
        print("ERROR:", e)
        tracker2.stop()
        raise e
    
    results = tokenizer.batch_decode(results, skip_special_tokens=True)
    new_batch_offset += 1

    return results, new_batch_offset

################################################################################################################

def data_thresholding(rawdata, preds, golds, score, thresh):
    comp_preds = []
    comp_golds = []
    rem_prompts = []
    rem_golds = []

    for raw, pr, gd, sc in zip(rawdata, preds, golds, score):
        if sc[1] > thresh:
            comp_preds.append(pr)
            comp_golds.append(gd)
        else:
            rem_prompts.append(raw)
            rem_golds.append(gd)

    return comp_preds, comp_golds, rem_prompts, rem_golds

################################################################################################################

def run_cascade(cascade_name, llm_chain, cascade_length, data_path, threshold = 0.8):
    rouge_score = evaluate.load('rouge')
    bert_score = evaluate.load("bertscore")
    
    data_name = data_path.split('/')[-1].split('.')[-2]

    ENERGY_OUT_DIR = ENERGY_OUT_DIR_BASE + f'{data_name}_{cascade_name}-{threshold}'
    if not os.path.exists(ENERGY_OUT_DIR):
        os.makedirs(ENERGY_OUT_DIR)
    else:
        os.system("rm " + os.path.join(ENERGY_OUT_DIR, "emissions.csv"))

    df = pd.read_csv(data_path)
    df = df.sort_values('prompt_text', key = lambda col: col.apply(len))
    raw_prompts = df['prompt_text'].values.tolist()
    raw_golds = df['summary'].apply(str.lower).values.tolist()
    # raw_prompts = raw_prompts[:5]
    # raw_golds = raw_golds[:5]

    final_preds = []
    final_golds = []

    batch_offset = 0
    for idx, llm in enumerate(llm_chain):
        print(f'\n{idx+1}. {llm}')
        data_loader = data_utils.DataLoader(raw_prompts, batch_size=4)
        model_name = llm.split('/')[-1]
        pred_data, batch_offset = run_llm(llm, data_loader, data_name, cascade_name, threshold, batch_offset, ENERGY_OUT_DIR)
        
        qa = pred_data.copy() # scorer copy

        if 'flan' not in model_name:
            pred_data = [pr[len(raw):] for pr, raw in zip(pred_data, raw_prompts)]
        
        if idx+1 != cascade_length:       # if current llm is not the last llm of the chain
            # qa = [query + str(ans) for query, ans in zip(raw_prompts, pred_data)]
            # log_file_path = '../models/samsum/' + cascade_name + '/log.csv'
            # log = pd.read_csv(log_file_path)
            scorer_path =  '../models/samsum/' + cascade_name + '/' + model_name
            score, batch_offset = get_score(qa, scorer_path, data_name, cascade_name, threshold, batch_offset, ENERGY_OUT_DIR)
            
            pred_data = map(str.lower, pred_data)
            pred_data = list(pred_data)

            # print(pred_data[:5])
            # print(qa[:5])
            # print(score[:5])
            # break

            comp_preds, comp_golds, rem_prompts, rem_golds = data_thresholding(raw_prompts, pred_data, raw_golds, score, threshold)
            final_preds.extend(comp_preds)
            final_golds.extend(comp_golds)

            raw_prompts = rem_prompts
            raw_golds = rem_golds

            print(f'Completed: {len(final_golds)}, Remaining: {len(rem_golds)}')

            if len(raw_prompts) == 0:
                break
        else:
            pred_data = map(str.lower, pred_data)
            pred_data = list(pred_data)

            final_preds.extend(pred_data)
            final_golds.extend(raw_golds)

    rg = rouge_score.compute(predictions=final_preds, references=final_golds)
    bt = bert_score.compute(predictions=final_preds, references=final_golds, model_type="distilbert-base-uncased")

    rouge1 = round(rg['rouge1'] * 100, 1)
    rouge2 = round(rg['rouge2'] * 100, 1)
    rougeL = round(rg['rougeL'] * 100, 1)
    prec = round(mean(bt['precision']),1)
    rec = round(mean(bt['recall']),1)
    f1 = round(mean(bt['f1']),1)

    print(f'Model: {cascade_name}, Threshold: {threshold}, Rouge1: {rouge1}, Rouge2: {rouge2}, RougeL: {rougeL}, Prec: {prec}, Rec: {rec}, F1: {f1}')
    return [cascade_name, threshold, rouge1, rouge2, rougeL, prec, rec, f1]

################################################################################################################


if __name__ == '__main__':

    # TRAINING

    # llm_chain = ['microsoft/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-small-8k-instruct', 'microsoft/Phi-3-medium-4k-instruct']
    # scorer1 = 'google/bigbird-roberta-base'
    # scorer2 = 'allenai/longformer-base-4096'

    # llm_chain = ['microsoft/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-small-8k-instruct', 'microsoft/Phi-3-medium-4k-instruct']
    # model_list = ['strategy-1', 'strategy-2', 'strategy-3', 'strategy-4']
    # scorer_list = [scorer1, scorer1, scorer2, scorer2]
    # lr_rate = [4e-5, 8e-5, 4e-5, 8e-5]
    # num_epochs = 8

    # for model, lr, scorer in zip(model_list, lr_rate, scorer_list):
    #     build_cascade(llm_chain, model, lr, num_epochs, scorer)

    # INFERENCE

    if not os.path.exists(PERF_OUT_DIR):
        os.makedirs(PERF_OUT_DIR)

    output_file_path = PERF_OUT_DIR + "samsum_st1-4.csv"
    if not os.path.exists(output_file_path):
        df = pd.DataFrame(columns = ["Model", "Threshold", "Rouge1", "Rouge2", "RougeL", "M-Pre", "M-Rec", "M-F1"])
        df.to_csv(output_file_path, index=False)

    llm_chain = ['microsoft/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-small-8k-instruct', 'microsoft/Phi-3-medium-4k-instruct']
    model_list = ['strategy-1', 'strategy-2', 'strategy-3', 'strategy-4']
    chain_list = [llm_chain, llm_chain, llm_chain, llm_chain]
    threshold = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    

    for model, llm_chain in zip(model_list, chain_list):
        for th in threshold:
            print(f'\nRunning samsum.py ---> Model: {model}, Threshold: {th}\n')
            perf = run_cascade(model, llm_chain, len(llm_chain), TEST_DATA_PATH, th)
            new_output = pd.DataFrame([perf], columns = ["Model", "Threshold", "Rouge1", "Rouge2", "RougeL", "M-Pre", "M-Rec", "M-F1"])

            prev_output = pd.read_csv(output_file_path)
            df = pd.concat([prev_output, new_output], ignore_index=True)
            df.to_csv(output_file_path, index=False)

    # df = pd.DataFrame(output, columns = ["Model", "Threshold", "M-Pre", "M-Rec", "M-F1"])
    # df.to_csv(os.path.join(PERF_OUT_DIR, "boolq_st7-8.csv"), index = False)
