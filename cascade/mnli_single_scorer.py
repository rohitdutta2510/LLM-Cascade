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
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from sklearn.metrics import f1_score, precision_score, recall_score

###############################################################################################################

PREDDIR = '../temp_outs/'
TRAIN_DATA_PATH = '../datasets/mnli_train_1k.csv'
TEST_DATA_PATH = '../datasets/mnli_test_1k.csv'
OUTDIR = '../performance/'
ENERGY_OUT_DIR_BASE = '../temp_outs/'

DEVICE = 'cuda:0'
TRACK_GPU = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else DEVICE[-1]

metric = evaluate.load("accuracy")

################################################################################################################

def get_data_mnli(datadir, preddir, modelname):
    rawdata = pd.read_csv(datadir)
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['label'].values.tolist()
    
    dn = preddir + '/mnli_train_1k_' + modelname + '/'
        
    try:
        with open(os.path.join(dn, "output.json")) as fp:
            preddata, timestamps = json.load(fp)
            if 'flan' not in modelname:
                preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
            preddata = map(str.lower, preddata) 

            pred = [1 if "1" in pr or "neutral" in pr else 2 if "2" in pr or "contradict" in pr or "contradiction" in pr else 0 for pr in preddata]

    except FileNotFoundError:
        print("Skipping", dn)     
        
    result = [1 if pred[i] == gold[i] else 0 for i in range(len(gold))]

    # print(gold[:10])
    # print(pred[:10])
    # print(result[:10])

    # qa = [query + str(ans) for query, ans in zip(rawdata.prompt_text, gold)]
    qa = [query + str(ans) for query, ans in zip(rawdata.prompt_text, pred)]
    # return {'query_answer' : qa, 'score' : result}
    return qa, result

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
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

################################################################################################################

def train(train_texts, train_labels, lr, epochs, save_dir = './mnli'):
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.6)
    # print("train_text 0",train_texts[0])
    # print("val_text 0",val_texts[0])

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True,max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True,max_length=512)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='../checkpoints',          # output directory
        num_train_epochs=epochs,              # total number of training epochs
        learning_rate=lr,  
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        # warmup_steps=500,                # number of warmup steps for learning rate scheduler
        # weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy ="epoch",
        load_best_model_at_end=True,
        # seed=42,
    )

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    
    model = model.to(DEVICE)
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset ,            # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)
        
################################################################################################################

def build_cascade(llm_chain, cascade_name = 'default', lr = 1e-5, epochs = 10):
    query_ans, result = [], []
    for model in llm_chain:
        qa, res = get_data_mnli(TRAIN_DATA_PATH, PREDDIR, model)
        query_ans.extend(qa)
        result.extend(res)


    save_path = os.path.join('../models/mnli_single_scorer', cascade_name)
    train(query_ans, result, lr, epochs, save_path)

    log_path = os.path.join('../models/mnli_single_scorer', cascade_name, 'log.csv')
    logs = pd.DataFrame([[lr, epochs]], columns = ["Learning rate", "Epochs"])
    logs.to_csv(log_path, index = False)

    print('\n\nBuilding LLM Cascade successful !!')

################################################################################################################

def get_score(text, scorer_path):
    model = DistilBertForSequenceClassification.from_pretrained(scorer_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = model.to(DEVICE)

    test_encodings = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
    test_encodings = {key: value.to(DEVICE) for key, value in test_encodings.items()}

    model.eval()

    dataset = data_utils.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
    dataloader = data_utils.DataLoader(dataset, batch_size=4)

    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            all_probs.extend(probs.cpu().numpy())
    
    return all_probs

################################################################################################################

def run_llm(model_path, data_loader, data_name, cascade_name, thresh, batch_offset, ENERGY_OUT_DIR, maxgentokens=50):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if 'flan-t5' in model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map = DEVICE, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map = DEVICE, torch_dtype=torch.bfloat16)

        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    results = []
    timestamps = []
    new_batch_offset = None

    try:
        for idx, batch in enumerate(tqdm(data_loader, ncols = 50)):
            bn = batch_offset + idx
            with OfflineEmissionsTracker(project_name="%s_single-scorer-%s-%s_%d"%(data_name, cascade_name, thresh, bn), experiment_id=bn, country_iso_code="IND", log_level="error",
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
    
    results = [tokenizer.batch_decode(results, skip_special_tokens=True), timestamps]
    torch.cuda.empty_cache()

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
    data_name = data_path.split('/')[-1].split('.')[-2]

    ENERGY_OUT_DIR = ENERGY_OUT_DIR_BASE + f'{data_name}_single-scorer-{cascade_name}-{threshold}'
    if not os.path.exists(ENERGY_OUT_DIR):
        os.makedirs(ENERGY_OUT_DIR)
    else:
        os.system("rm " + os.path.join(ENERGY_OUT_DIR, "emissions.csv"))
    
    # print('Cascade Length: ', cascade_length)
    df = pd.read_csv(data_path)
    df = df.sort_values('prompt_text', key = lambda col: col.apply(len))

    raw_prompts = df['prompt_text'].values.tolist()
    raw_golds = df['label'].values.tolist()
    
    final_preds = []
    final_golds = []

    batch_offset = 0
    for idx, llm in enumerate(llm_chain):
        print(f'\n{idx+1}. {llm}')
        data_loader = data_utils.DataLoader(raw_prompts, batch_size=16)
        model_name = llm.split('/')[-1]
        response, batch_offset = run_llm(llm, data_loader, data_name, cascade_name, threshold, batch_offset, ENERGY_OUT_DIR)

        if 'flan' not in model_name:
            pred_data = [pr[len(raw):] for pr, raw in zip(response[0], raw_prompts)]
        pred_data = map(str.lower, response[0]) 
        pred_data = [1 if "1" in pr or "neutral" in pr else 2 if "2" in pr or "contradict" in pr or "contradiction" in pr else 0 for pr in pred_data]

        if idx+1 != cascade_length:       # if current llm is not the last llm of the chain
            qa = [query + str(ans) for query, ans in zip(raw_prompts, pred_data)] 
            score = get_score(qa, '../models/mnli_single_scorer/' + cascade_name)

            # print(response[0][:5])
            # print(pred_data[:5])
            # print(qa[:5])
            # print(score[:5])
            
            comp_preds, comp_golds, rem_prompts, rem_golds = data_thresholding(raw_prompts, pred_data, raw_golds, score, threshold)
            final_preds.extend(comp_preds)
            final_golds.extend(comp_golds)

            raw_prompts = rem_prompts
            raw_golds = rem_golds
            batch_offset += 1
            
            print(f'Completed: {len(final_golds)}, Remaining: {len(rem_golds)}')

            if len(raw_prompts) == 0:
                break
        else:
            final_preds.extend(pred_data)
            final_golds.extend(raw_golds)

    prec = round(precision_score(final_golds, final_preds, average="macro") * 100, 1)
    rec = round(recall_score(final_golds, final_preds, average="macro") * 100, 1)
    f1 = round(f1_score(final_golds, final_preds, average="macro") * 100, 1)

    print(f'\nPrecision: {prec}, Recall: {rec}, F1-Score: {f1}\n')

    return [cascade_name, threshold, prec, rec, f1]

################################################################################################################


if __name__ == '__main__':

    # TRAINING

    llm_chain = ['flan-t5-base', 'flan-t5-large', 'flan-t5-xl']
    model_list = ['strategy_1', 'strategy_2', 'strategy_3', 'strategy_4']
    lr_rate = [1e-5, 2e-5, 5e-5, 1e-4]
    num_epochs = 10

    for model, lr in zip(model_list, lr_rate):
        build_cascade(llm_chain, model, lr, num_epochs)

    # INFERENCE

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    chain_1 = ['google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl']
    model_list = ['strategy_1', 'strategy_2', 'strategy_3', 'strategy_4']
    chain_list = [chain_1, chain_1, chain_1, chain_1, chain_1]
    threshold = [0.9, 0.95]

    output = []
    for model, llm_chain in zip(model_list, chain_list):
        for th in threshold:
            output.append(run_cascade(model, llm_chain, len(llm_chain), TEST_DATA_PATH, th))

    df = pd.DataFrame(output, columns = ["Model", "Threshold", "M-Pre", "M-Rec", "M-F1"])
    df.to_csv(os.path.join(OUTDIR, "mnli_single_scorer.csv"), index = False)
