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

# INDIR = '../llm-sust/datasets/datasets_train_256/'
PREDDIR = '../llm-sust/temp_outs/temp_outs_cascade/'
TRAIN_DATA_PATH = '../llm-sust/datasets/datasets_llm_cascade/boolq_train_1k.csv'
TEST_DATA_PATH = '../llm-sust/datasets/datasets_llm_cascade/boolq_test_1k.csv'

DEVICE = 'cuda:0'

metric = evaluate.load("accuracy")

################################################################################################################

def get_data_boolq(datadir, preddir, modelname):
    rawdata = pd.read_csv(datadir)
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['label'].values.tolist()
    
    dn = preddir + '/boolq_train_1k_' + modelname + '/'
        
    try:
        with open(os.path.join(dn, "output.json")) as fp:
            preddata, timestamps = json.load(fp)
            if 'flan' not in modelname:
                preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
            preddata = map(str.lower, preddata) 

            pred = [1 if "true" in pr or "yes" in pr else 0 for pr in preddata]

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

def train(train_texts, train_labels, save_dir = './boolq'):
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.6)
    # print("train_text 0",train_texts[0])
    # print("val_text 0",val_texts[0])

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True,max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True,max_length=512)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='./scorer_location',          # output directory
        num_train_epochs=10,              # total number of training epochs
        learning_rate = 5e-5,               # initial learning rate
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        # warmup_steps=500,                # number of warmup steps for learning rate scheduler
        # weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy ="epoch",
        load_best_model_at_end=True,
        seed=42,
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

def build_cascade(llm_chain = ['flan-t5-base', 'flan-t5-large'], cascade_name = 'default'):
    query_ans, result = [], [] 
    for model in llm_chain:
        qa, res = get_data_boolq(TRAIN_DATA_PATH, PREDDIR, model)
        query_ans.extend(qa)
        result.extend(res)
    
    print('No. of training data: ', len(query_ans))
    save_path = './boolq_single_scorer/' + cascade_name
    train(query_ans, result, save_path)

    print('\n\nBuilding LLM Cascade successful !!')

################################################################################################################

def get_score(text, scorer_path):
    model = DistilBertForSequenceClassification.from_pretrained(scorer_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = model.to(DEVICE)

    test_encodings = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
    test_encodings = {key: value.to(DEVICE) for key, value in test_encodings.items()}

    model.eval()

    # Create TensorDataset and DataLoader
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

    # with torch.no_grad():
    #     outputs = model(**test_encodings)
    #     probs = F.softmax(outputs.logits, dim=-1)
    #     # predictions = torch.argmax(outputs.logits, dim=-1)

    # # print("Outputs:", outputs)
    # # print("Probabilities:", probs.cpu().numpy())
    # # print("Predictions:", predictions)
    # return probs.cpu().numpy()

################################################################################################################

def run_llm(model_path, data_loader, MAXGENTOKENS=50):
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

    try:
        for idx, batch in enumerate(tqdm(data_loader, ncols = 50)):
            
            st = time()

            batchdata = tokenizer(batch, return_tensors="pt", padding = True, truncation =  True)
            inp = batchdata.input_ids.to(DEVICE)
            attn = batchdata.attention_mask.to(DEVICE)
            
            outputs = model.generate(inp, attention_mask=attn, max_new_tokens=MAXGENTOKENS, pad_token_id=tokenizer.pad_token_id)

            end = time()
            results.extend(outputs)
            timestamps.append((st,end))

            gc.collect()
            torch.cuda.empty_cache()

    except Exception as e:
        print("ERROR:", e)
        raise e
    
    results = [tokenizer.batch_decode(results, skip_special_tokens=True), timestamps]
    torch.cuda.empty_cache()

    return results

################################################################################################################

def data_thresholding(rawdata, preds, score, thresh = 0.9):
    comp_prompts = []
    comp_preds = []
    rem_prompts = []
    # rem_preds = []
    for raw, pr, sc in zip(rawdata, preds, score):
        if sc[0] > thresh or sc[1] > thresh:
            comp_prompts.append(raw)
            comp_preds.append(pr)
        else:
            rem_prompts.append(raw)
            # rem_preds.append(pr)

    return comp_prompts, comp_preds, rem_prompts

################################################################################################################

def run_cascade(cascade_name, llm_chain, cascade_length, data_path):
    # print('Cascade Length: ', cascade_length)

    df = pd.read_csv(data_path)
    df = df.sort_values('prompt_text', key = lambda col: col.apply(len))
    rawdata = df['prompt_text'].values.tolist()
    gold = df['label'].values.tolist()
    
    prompts = []
    preds = []

    for idx, llm in enumerate(llm_chain):
        print(idx+1, llm)
        data_loader = data_utils.DataLoader(rawdata, batch_size=4)
        model_name = llm.split('/')[-1]
        response = run_llm(llm, data_loader)
        
        if 'flan' not in model_name:
            preddata = [pr[len(raw):] for pr, raw in zip(response[0], rawdata)]
        preddata = map(str.lower, response[0]) 
        preddata = [1 if "true" in pr or "yes" in pr else 0 for pr in preddata] 

        torch.cuda.empty_cache()

        if idx+1 != cascade_length:       # if current llm is not the last llm of the chain
            qa = [query + str(ans) for query, ans in zip(rawdata, preddata)] 
            score = get_score(qa, './boolq_single_scorer/' + cascade_name)

            # print(response[0][:5])
            # print(pred[:5])
            # print(qa[:5])
            # print(score[:5])

            comp_prompts, comp_preds, rem_prompts = data_thresholding(rawdata, preddata, score, 0.9)
            prompts.extend(comp_prompts)
            preds.extend(comp_preds)
            rawdata = rem_prompts
            print(len(prompts), len(preds), len(rawdata))

            if len(rawdata) == 0:
                break
        else:
            prompts.extend(rawdata)
            preds.extend(preddata)

        torch.cuda.empty_cache()
    
    final_data = pd.DataFrame({'prompt_text':prompts, 'preds': preds})
    final_data = final_data.sort_values('prompt_text', key = lambda col: col.apply(len))

    # check = pd.DataFrame({'prompts' : final_data['prompt_text'].values.tolist(), 'preds' : final_data['preds'].values.tolist(), 'gold' : gold})
    # check.to_csv('check.csv')
    
    prec = round(precision_score(gold, final_data['preds'].values.tolist(), average="macro") * 100, 1)
    rec = round(recall_score(gold, final_data['preds'].values.tolist(), average="macro") * 100, 1)
    f1 = round(f1_score(gold, final_data['preds'].values.tolist(), average="macro") * 100, 1)

    print('\n')
    print(f'Precision: {prec}, Recall: {rec}, F1-Score: {f1}')

################################################################################################################


if __name__ == '__main__':

    # llm_chain = ['flan-t5-base', 'flan-t5-large', 'flan-t5-xl']
    # chain_name = 'strategy_1'
    # build_cascade(llm_chain, chain_name)
    
    llm_chain = ['google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl']
    chain_name = 'strategy_1'
    run_cascade(chain_name, llm_chain, len(llm_chain), TEST_DATA_PATH)
