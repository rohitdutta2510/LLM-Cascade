import json
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, root_mean_squared_error, mean_absolute_error, r2_score
import evaluate
import math
from statistics import mean


# datasets = ['boolq', 'copa', 'qnli', 'cola', 'stsb', 'sst2', 'mnli', 'mrpc', 'wnli', 'cnn_dm']
# datasets = ['cola', 'qnli', 'wnli', 'mnli']
# datasets = ['boolq', 'sst2']
datasets = ['boolq']
# datasets = ['boolq', 'vax', 'caves', 'sst2', 'stsb', 'copa', 'cola', 'mrpc', 'mnli', 'qnli', 'samsum', 'dolly', 'hatex']


INDIR = '../datasets/'
PREDDIR = '../temp_outs/'
OUTDIR = '../performance_dataset_wise/'
# PREDDIR = '../../temp_outs/temp_outs_pc_16/'
# OUTDIR = '../../results/performance_dataset_wise_ft_16/'


# INDIR = '../../datasets/datasets_train_4k/'
# PREDDIR = '../../temp_outs/temp_outs_16_fix_output/'
# OUTDIR = '../../results/performance_energy_prompt/energy_prompt_16_fix_output'


#########################################################################################
#########################################################################################
def eval_cnn_dm(datadir, preddir, outdir):
    metrics = evaluate.load('rouge')
    bertscore = evaluate.load("bertscore")

    rawdata = pd.read_csv(os.path.join(datadir, "cnn_dm.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['highlights'].apply(str.lower).values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'cnn_dm' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
           
            if 'flan' not in modelname:
                preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
            preddata = map(str.lower, preddata) 

            pred = list(preddata)

        except FileNotFoundError:
            print("Skipping", dn)
            continue
            
            
        results = metrics.compute(predictions=pred, references=gold)
        bert_score = bertscore.compute(predictions=pred, references=gold, model_type="distilbert-base-uncased")
        # print(modelname, results)

        outdata.append([modelname])
        outdata[-1].append(round(results['rouge1'] * 100, 1))
        outdata[-1].append(round(results['rouge2'] * 100, 1))
        outdata[-1].append(round(results['rougeL'] * 100, 1))
        outdata[-1].append(round(mean(bert_score['precision']),1))
        outdata[-1].append(round(mean(bert_score['recall']),1))
        outdata[-1].append(round(mean(bert_score['f1']),1))
        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns = ["Model", "ROUGE1", "ROUGE2", "ROUGEL", "Prec", "Rec", "F1"])
    df2.to_csv(os.path.join(outdir, "cnn_dm.csv"), index = False)



#########################################################################################
#########################################################################################
def eval_samsum(datadir, preddir, outdir):
    metrics = evaluate.load('rouge')
    bertscore = evaluate.load("bertscore")

    rawdata = pd.read_csv(os.path.join(datadir, "samsum.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['summary'].apply(str.lower).values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'samsum' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
           
            if 'flan' not in modelname:
                preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
            preddata = map(str.lower, preddata) 

            pred = list(preddata)

        except FileNotFoundError:
            print("Skipping", dn)
            continue
            
            
        results = metrics.compute(predictions=pred, references=gold)
        bert_score = bertscore.compute(predictions=pred, references=gold, model_type="distilbert-base-uncased")
        # print(modelname, results)
        # print(mean(bert_score['f1']), '\n\n')

        outdata.append([modelname])
        outdata[-1].append(round(results['rouge1'] * 100, 1))
        outdata[-1].append(round(results['rouge2'] * 100, 1))
        outdata[-1].append(round(results['rougeL'] * 100, 1))
        outdata[-1].append(round(mean(bert_score['precision']),1))
        outdata[-1].append(round(mean(bert_score['recall']),1))
        outdata[-1].append(round(mean(bert_score['f1']),1))
        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns = ["Model", "ROUGE1", "ROUGE2", "ROUGEL", "Prec", "Rec", "F1"])
    df2.to_csv(os.path.join(outdir, "samsum.csv"), index = False)



#########################################################################################
#########################################################################################
def eval_qnli(datadir, preddir, outdir):
    rawdata = pd.read_csv(os.path.join(datadir, "qnli.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['label'].values.tolist()


    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'qnli' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata) 

                # pred = [1 if "1" in pr or "not entail" in pr or "not entailment" in pr or "not entailed" in pr or "does not entail" in pr else 0 for pr in preddata]
                pred = [1 if "not entail" in pr or "not entailment" in pr or "not entailed" in pr or "does not entail" in pr else 0 for pr in preddata]
                # print(pred[:20])

        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])
        
    df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "qnli.csv"), index = False)



#########################################################################################
#########################################################################################
def eval_wnli(datadir, preddir, outdir):
    rawdata = pd.read_csv(os.path.join(datadir, "wnli.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['label'].values.tolist()


    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'wnli' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata) 

                pred = [0 if "0" in pr or "not entail" in pr else 1 for pr in preddata]
                # print(pred[:20])

        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])
        
    df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "wnli.csv"), index = False)




#########################################################################################
#########################################################################################
def eval_mnli(datadir, preddir, outdir):
    rawdata = pd.read_csv(os.path.join(datadir, "mnli_test_1k.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['label'].values.tolist()


    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'mnli_test_1k' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata) 

                # pred = []
                # for pr in preddata:
                #     if "1" in pr or "neutral" in pr:
                #         pred.append(1)
                #     elif "2" in pr or "contradict" in pr or "contradiction" in pr:
                #         pred.append(2)
                #     else:
                #         pred.append(0)
                pred = [1 if "1" in pr or "neutral" in pr else 2 if "2" in pr or "contradict" in pr or "contradiction" in pr else 0 for pr in preddata]

                # pred = [1 if "1" in pr or "neutral" in pr else 2 if "2" in pr or 'contradict' in pr else 0 for pr in preddata]
                # print(pred[:20])

        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro", zero_division=0) * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro", zero_division=0) * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro", zero_division=0) * 100, 1))

        print(outdata[-1])
        
    # df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    # df2.to_csv(os.path.join(outdir, "mnli.csv"), index = False)



#########################################################################################
#########################################################################################
def eval_cola(datadir, preddir, outdir):
    rawdata = pd.read_csv(os.path.join(datadir, "cola.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['label'].values.tolist()


    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'cola' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata) 

                pred = [0 if "0" in pr or "not accept" in pr or "not acceptable" in pr else 1 for pr in preddata]
                # print(preddata[:20])

        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])
        
    df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "cola.csv"), index = False)



#########################################################################################
#########################################################################################
def eval_sst2(datadir, preddir, outdir):
    rawdata = pd.read_csv(os.path.join(datadir, "sst2.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['label'].values.tolist()


    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'sst2' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata) 

                pred = [1 if "1" in pr or "positive" in pr else 0 for pr in preddata]
                # print(preddata[:20])

        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])
        
    df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "sst2.csv"), index = False)





#########################################################################################
#########################################################################################
def eval_boolq(datadir, preddir, outdir):
    rawdata = pd.read_csv(os.path.join(datadir, "boolq_test_1k.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['label'].values.tolist()


    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'boolq_test_1k' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata) 

                pred = [1 if "true" in pr or "yes" in pr else 0 for pr in preddata]
                # print(preddata[:5])

        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])
        
    # df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    # df2.to_csv(os.path.join(outdir, "boolq.csv"), index = False)

#########################################################################################
#########################################################################################
def eval_mrpc(datadir, preddir, outdir):
    rawdata = pd.read_csv(os.path.join(datadir, "mrpc.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['label'].values.tolist()


    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'mrpc' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata) 

                pred = [0 if "0" in pr else 1 for pr in preddata]
                # print(pred[:5])

        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])
        
    df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "mrpc.csv"), index = False)

#########################################################################################
#########################################################################################
def eval_copa(datadir, preddir, outdir):
    rawdata = pd.read_csv(os.path.join(datadir, "copa.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['label'].values.tolist()


    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'copa' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata) 

                pred = [0 if "0" in pr or "choice1" in pr else 1 for pr in preddata]
                # print(pred[:20])

        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])
        
    df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "copa.csv"), index = False)



#########################################################################################
#########################################################################################
def eval_vax(datadir, preddir, outdir):
    gold_label_list = ['ProVax', 'AntiVax', 'Neutral']
    pred_label_list = ['pro-vaccine', 'anti-vaccine', 'neutral']
    
    rawdata = pd.read_csv(os.path.join(datadir, "vax.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    golddata = rawdata['majority_label'].values.tolist()
    golddata = ['All' if isinstance(x, float) and math.isnan(x) else x for x in golddata]
    gold = [[int(lab in row) for lab in gold_label_list] for row in golddata]
    # print(gold[:5])

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'vax' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                    # print(preddata[:5])
                preddata = map(str.lower, preddata) 

                pred = [[int(lab in row) for lab in pred_label_list] for row in preddata]
                # pred = [0 if "0" in pr or "choice1" in pr else 1 for pr in preddata]
                # print(pred[:5], '\n\n')

        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])
        
    df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "vax.csv"), index = False)


#########################################################################################
#########################################################################################
def eval_caves(datadir, preddir, outdir):
    labellist = ["unnecessary", "mandatory", "pharma", "conspiracy", "political", "country", "rushed", "ingredients", "side-effect", "ineffective", "religious", "none"]
    
    rawdata = pd.read_csv(os.path.join(datadir, "caves.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    golddata = rawdata['labels'].values.tolist()
    # print(golddata[:5])
    gold = [[int(lab in row)  for lab in labellist] for row in golddata]
    # print('gold : ', gold[:1], '\n\n')

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'caves' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = list(map(str.lower, preddata))
                # print(preddata[:5])
                # pred = [0 if "0" in pr or "choice1" in pr else 1 for pr in preddata]
                pred = [[int(lab in row)  for lab in labellist] for row in preddata]
                # print(pred[:5], '\n\n')


        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro", zero_division=0) * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro", zero_division=0) * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro", zero_division=0) * 100, 1))

        print(outdata[-1])
  
    df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "caves.csv"), index = False)

#########################################################################################
#########################################################################################
def eval_stsb(datadir, preddir, outdir):    
    rawdata = pd.read_csv(os.path.join(datadir, "stsb.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['label'].values.tolist()
    # print(gold[:5])

    outdata = []
    num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'stsb' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                # preddata = map(str.lower, preddata) 
                score = [s[:3] for s in preddata]
                # score_final = []
                # for s in score:
                #     if s[1] == '/': score_final.append(s[0])
                #     elif s[1] == '.' and s[2] in num: score_final.append(s)
                #     else: score_final.append(s[0])
                score = [float(s[0]) if s[1] == '/' or not (s[1] == '.' and s[2] in num) else float(s) for s in score]
                # print(score_final[:5])

        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(root_mean_squared_error(gold, score), 2))
        outdata[-1].append(round(mean_absolute_error(gold, score), 2))
        outdata[-1].append(round(r2_score(gold, score), 2))

        print(outdata[-1])
        
    df2 = pd.DataFrame(outdata, columns = ["Model", "RMSE", "MAE", "R2"])
    df2.to_csv(os.path.join(outdir, "stsb.csv"), index = False)

#########################################################################################
#########################################################################################
def eval_dolly(datadir, preddir, outdir):
    metrics = evaluate.load('rouge')
    bertscore = evaluate.load("bertscore")
    
    rawdata = pd.read_csv(os.path.join(datadir, "dolly-hhrlhf.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    gold = rawdata['response'].apply(str.lower).values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'dolly-hhrlhf' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
           
            if 'flan' not in modelname:
                preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
            preddata = map(str.lower, preddata) 

            pred = list(preddata)

        except FileNotFoundError:
            print("Skipping", dn)
            continue
            
            
        results = metrics.compute(predictions=pred, references=gold)
        bert_score = bertscore.compute(predictions=pred, references=gold, model_type="distilbert-base-uncased")
        # print(modelname, results)

        outdata.append([modelname])
        outdata[-1].append(round(results['rouge1'] * 100, 1))
        outdata[-1].append(round(results['rouge2'] * 100, 1))
        outdata[-1].append(round(results['rougeL'] * 100, 1))
        outdata[-1].append(round(mean(bert_score['precision']),1))
        outdata[-1].append(round(mean(bert_score['recall']),1))
        outdata[-1].append(round(mean(bert_score['f1']),1))
        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns = ["Model", "ROUGE1", "ROUGE2", "ROUGEL", "Prec", "Rec", "F1"])
    df2.to_csv(os.path.join(outdir, "dolly-hhrlhf.csv"), index = False)

#########################################################################################
#########################################################################################
def eval_hatex(datadir, preddir, outdir):
    true_label = ['hatespeech', 'offensive', 'normal']
    pred_label1 = ['hate speech', 'hatespeech']
    pred_label2 = ['offensive']
    pred_label3 = ['normal']
    
    rawdata = pd.read_csv(os.path.join(datadir, "hatexplain.csv"))
    rawdata = rawdata.sort_values('prompt_text', key = lambda col: col.apply(len))
    golddata = rawdata['label'].values.tolist()
    gold = [[int(lab in row)  for lab in true_label] for row in golddata]
    # print(gold[:5])
    
    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'hatexplain' in dn]):       
        modelname = dn[dn.rfind('_') + 1:]
        
        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                    # print(preddata[:5])
                preddata = map(str.lower, preddata) 
                preddata = [p[:15] for p in preddata]
   
                pred = []
                for pr in preddata:
                    hate = any(lab in pr for lab in pred_label1)
                    off = any(lab in pr for lab in pred_label2)
                    norm = any(lab in pr for lab in pred_label3)
                    
                    pred.append([int(hate), int(off), int(norm)])
                # print(pred[:5])

        except FileNotFoundError:
            print("Skipping", dn)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])
        
    df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "hatex.csv"), index = False)


#########################################################################################
#########################################################################################
#########################################################################################
if __name__ == '__main__':
    # if not os.path.exists(OUTDIR):
    #     os.makedirs(OUTDIR)


    for name in datasets:
        print("\n\n" + "#" * 50 + " " + name)
        func = eval("eval_" + name)
        func(INDIR, PREDDIR, OUTDIR)
    
    